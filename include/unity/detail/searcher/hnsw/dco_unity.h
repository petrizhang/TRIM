/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#ifdef USE_AVX
#include <immintrin.h>
#endif

#include "faiss/impl/code_distance/code_distance.h"
#include "unity/common/atomic.h"
#include "unity/common/dco.h"
#include "unity/common/object.h"
#include "unity/common/prefetch.h"
#include "unity/common/setter_proxy.h"
#include "unity/detail/index/hnsw.h"

namespace unity {
namespace detail {

template <typename PQDecoderType, bool enable_profile = false>
struct UnityOp final : SetterProxy<UnityOp<PQDecoderType, enable_profile>>,
                       IDistanceComparisonOperator<unsigned, float> {
  using This = UnityOp<PQDecoderType, enable_profile>;
  using Proxy = SetterProxy<This>;
  using Parent = IDistanceComparisonOperator<unsigned, float>;
  using idx_t = unsigned;
  using dist_t = float;

  // Data structures for query
  const dist_t* dist_table_data{nullptr};
  const uint8_t* codes;
  const float* recons_errors;
  size_t M{0};
  size_t nbits{0};
  size_t code_size{0};
  hnswlib::DISTFUNC<dist_t> dist_func{nullptr};
  void* dist_func_param{nullptr};

  // Indexes
  const faiss::IndexPQ* faiss_index_pq{nullptr};
  const hnswlib::HierarchicalNSW<float>* hnsw{nullptr};
  const dist_t* query{nullptr};
  faiss::AlignedTable<dist_t> dist_table;

  // Profile metrics
  Atomic<int64_t> num_distance_computation;
  Atomic<int64_t> num_pq_distance_computation;
  Atomic<int64_t> num_lowerbound_computation;

  // Search parameters
  float _gamma{0.8};

  ~UnityOp() override = default;

  explicit UnityOp(const UnityHnsw* p_uhnsw) : SetterProxy<This>("UnityOp") {
    U_ASSERT(p_uhnsw != nullptr);
    U_ASSERT(p_uhnsw->owned_index_hnsw != nullptr);
    U_ASSERT(p_uhnsw->unity_index_pq.owned_index_pq != nullptr);
    faiss_index_pq = p_uhnsw->unity_index_pq.owned_index_pq.get();
    M = faiss_index_pq->pq.M;
    nbits = faiss_index_pq->pq.nbits;
    code_size = faiss_index_pq->code_size;
    codes = p_uhnsw->unity_index_pq.codes.data();
    recons_errors = p_uhnsw->unity_index_pq.recons_errors.data();

    hnsw = p_uhnsw->owned_index_hnsw.get();
    dist_func = hnsw->fstdistfunc_;
    dist_func_param = hnsw->dist_func_param_;

    Proxy::template bind<DOUBLE_TYPE>("gamma", &This::set_gamma);
  }

  void set_query(const dist_t* query_data) override {
    query = query_data;
    dist_table.resize(faiss_index_pq->pq.M * faiss_index_pq->pq.ksub);
    faiss_index_pq->pq.compute_distance_table(query, dist_table.data());
    dist_table_data = dist_table.data();
  }

  void set(const std::string& key, const Object& value) override { Proxy::proxy_set(key, value); }

  void try_set(const std::string& key, const Object& value) { Proxy::proxy_try_set(key, value); }

  bool dist_comp(dist_t max_dist, idx_t i, float& dist) const override final {
    dist_t lowerbound = relaxed_lowerbound(i);
    if (lowerbound < max_dist) {
      dist = compute(i);
      return dist < max_dist;
    }
    return false;
  }

  void dist_comp8(dist_t max_dist, const Id8& ids, Dist8& dists,
                  Bool8& lt_flags) const override final {
    _dist_comp8(max_dist, ids, dists, lt_flags);
  }

  dist_t compute(idx_t i) const override {
    assert(query != nullptr);
    return dist_func(query, hnsw->getDataByInternalId(i), dist_func_param);
  }

  dist_t relaxed_lowerbound(idx_t i) const override {
    assert(query != nullptr);
    dist_t a = std::sqrt(estimate(i));
    dist_t b = recons_errors[i];
    return (a - b) * (a - b) + 2 * _gamma * a * b;
  }

  void relaxed_lowerbound8(const Id8& ids, Dist8& dists) const override {
    assert(query != nullptr);
    _relaxed_lowerbound8(ids, dists);
  }

  dist_t estimate(idx_t i) const override {
    return faiss::distance_single_code<PQDecoderType>(M, nbits, dist_table_data,
                                                      codes + i * code_size);
  }

  void prefetch(idx_t i) const override { prefetch_l1(codes + code_size * i); }

  void set_gamma(float gamma) { _gamma = gamma; }

  void _prefetch_vector(idx_t i) const { prefetch_l1(hnsw->getDataByInternalId(i)); }

#ifdef USE_AVX
  void _relaxed_lowerbound8(const Id8& ids, Dist8& dists) const {
    // PQ distances
    float pq_dist[8] = {
        0,
    };

    prefetch_l1(codes + ids[4] * code_size);
    prefetch_l1(codes + ids[5] * code_size);
    prefetch_l1(codes + ids[6] * code_size);
    prefetch_l1(codes + ids[7] * code_size);

    faiss::distance_four_codes<PQDecoderType>(
        M, nbits, dist_table_data,                                //
        codes + ids[0] * code_size, codes + ids[1] * code_size,  //
        codes + ids[2] * code_size, codes + ids[3] * code_size,  //
        pq_dist[0], pq_dist[1], pq_dist[2], pq_dist[3]);

    // Prefetch reconstruciton errors
    prefetch_l1(recons_errors + ids[0]);
    prefetch_l1(recons_errors + ids[1]);
    prefetch_l1(recons_errors + ids[2]);
    prefetch_l1(recons_errors + ids[3]);
    prefetch_l1(recons_errors + ids[4]);
    prefetch_l1(recons_errors + ids[5]);
    prefetch_l1(recons_errors + ids[6]);
    prefetch_l1(recons_errors + ids[7]);

    faiss::distance_four_codes<PQDecoderType>(
        M, nbits, dist_table_data,                                //
        codes + ids[4] * code_size, codes + ids[5] * code_size,  //
        codes + ids[6] * code_size, codes + ids[7] * code_size,  //
        pq_dist[4], pq_dist[5], pq_dist[6], pq_dist[7]);

    // PQ distances
    __m256 vec_pq_dist = _mm256_loadu_ps(pq_dist);
    __m256 vec_recons_error = _mm256_set_ps(recons_errors[ids[7]], recons_errors[ids[6]],  //
                                            recons_errors[ids[5]], recons_errors[ids[4]],  //
                                            recons_errors[ids[3]], recons_errors[ids[2]],  //
                                            recons_errors[ids[1]], recons_errors[ids[0]]);
    // Lowerbounds
    __m256 vec_lowerbounds = _relaxed_lowerbound8_avx2(_gamma, vec_pq_dist, vec_recons_error);
    _mm256_storeu_ps(dists.data(), vec_lowerbounds);
  }

  void _dist_comp8(dist_t max_dist, const Id8& ids, Dist8& dists, Bool8& lt_flags) const {
    // PQ distances
    float pq_dist[8] = {
        0,
    };

    prefetch_l1(codes + ids[4] * code_size);
    prefetch_l1(codes + ids[5] * code_size);
    prefetch_l1(codes + ids[6] * code_size);
    prefetch_l1(codes + ids[7] * code_size);

    faiss::distance_four_codes<PQDecoderType>(
        M, nbits, dist_table_data,                                //
        codes + ids[0] * code_size, codes + ids[1] * code_size,  //
        codes + ids[2] * code_size, codes + ids[3] * code_size,  //
        pq_dist[0], pq_dist[1], pq_dist[2], pq_dist[3]);

    // Prefetch reconstruciton errors
    prefetch_l1(recons_errors + ids[0]);
    prefetch_l1(recons_errors + ids[1]);
    prefetch_l1(recons_errors + ids[2]);
    prefetch_l1(recons_errors + ids[3]);
    prefetch_l1(recons_errors + ids[4]);
    prefetch_l1(recons_errors + ids[5]);
    prefetch_l1(recons_errors + ids[6]);
    prefetch_l1(recons_errors + ids[7]);

    faiss::distance_four_codes<PQDecoderType>(
        M, nbits, dist_table_data,                                //
        codes + ids[4] * code_size, codes + ids[5] * code_size,  //
        codes + ids[6] * code_size, codes + ids[7] * code_size,  //
        pq_dist[4], pq_dist[5], pq_dist[6], pq_dist[7]);

    // PQ distances
    __m256 vec_pq_dist = _mm256_loadu_ps(pq_dist);
    __m256 vec_recons_error = _mm256_set_ps(recons_errors[ids[7]], recons_errors[ids[6]],  //
                                            recons_errors[ids[5]], recons_errors[ids[4]],  //
                                            recons_errors[ids[3]], recons_errors[ids[2]],  //
                                            recons_errors[ids[1]], recons_errors[ids[0]]);
    // Lowerbounds
    __m256 vec_lowerbounds = _relaxed_lowerbound8_avx2(_gamma, vec_pq_dist, vec_recons_error);
    __m256 cmp_vec = _mm256_cmp_ps(vec_lowerbounds, _mm256_set1_ps(max_dist), _CMP_LT_OS);
    lt_flags.mask = _mm256_movemask_ps(cmp_vec);

    if (!lt_flags.has_true()) {
      return;
    }

    unsigned qual_size = 0;
    unsigned qual_ids[8];
    unsigned qual_pos[8];
    for (unsigned i = 0; i < 8; i++) {
      if (lt_flags.get(i)) {
        qual_ids[qual_size] = ids[i];
        qual_pos[qual_size] = i;
        ++qual_size;
      }
    }

    unsigned i = 0;
    for (; i < qual_size - 1; i++) {
      _prefetch_vector(qual_ids[i + 1]);
      dist_t dist = compute(qual_ids[i]);
      dists[qual_pos[i]] = dist;
    }

    dist_t dist = compute(qual_ids[i]);
    dists[qual_pos[i]] = dist;
  }

  static __m256 _relaxed_lowerbound8_avx2(float gamma, __m256 pq_dist_vec,
                                          __m256 recons_error_vec) {
    // Squared root of PQ distances
    __m256 vec_a = _mm256_sqrt_ps(pq_dist_vec);
    // Reconstruction errors (actully squared roots of reconstruction errors)
    __m256 vec_b = recons_error_vec;
    // lowerbounds = (a[i] - b[i])^2 + 2 * gamma * a[i] * b[i]
    __m256 vec_diff = _mm256_sub_ps(vec_a, vec_b);
    __m256 vec_diff_squared = _mm256_mul_ps(vec_diff, vec_diff);
    __m256 vec_2gamma = _mm256_set1_ps(2 * gamma);
    __m256 vec_ab = _mm256_mul_ps(vec_a, vec_b);
    __m256 vec_2gamma_ab = _mm256_mul_ps(vec_2gamma, vec_ab);
    __m256 lowerbounds = _mm256_add_ps(vec_diff_squared, vec_2gamma_ab);
    return lowerbounds;
  }
#else
  void _relaxed_lowerbound8(const Id8& ids, Dist8& dists) const {
    Parent::relaxed_lowerbound8(ids, dists);
  }

  void _dist_comp8(dist_t max_dist, const Id8& ids, Dist8& dists) const {
    Parent::dist_comp8(max_dist, ids, dists);
  }
#endif
};

template <bool enable_profile = false>
using UnityOp8 = UnityOp<faiss::PQDecoder8, enable_profile>;

template <typename T>
constexpr const bool is_unity_dco_v = false;

template <typename T, bool v>
constexpr const bool is_unity_dco_v<UnityOp<T, v>> = true;

}  // namespace detail
}  // namespace unity