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

#include "unity/common/atomic.h"
#include "unity/common/dco.h"
#include "unity/common/prefetch.h"
#include "unity/detail/faiss/impl/ProductQuantizer.h"
#include "unity/detail/faiss/impl/code_distance/code_distance.h"
#include "unity/detail/hnswlib/hnswlib.h"
#include "unity/detail/uhnsw/unity_hnsw.h"

namespace unity {
namespace detail {

template <typename PQDecoderType, bool enable_profile = false>
struct UnityOp final : IDistanceComparisonOperator<unsigned, float> {
  using Parent = IDistanceComparisonOperator<unsigned, float>;
  using idx_t = unsigned;
  using dist_t = float;

  // Data structures for query
  const dist_t* _dist_table_data{nullptr};
  const uint8_t* _codes;
  const float* _recons_errors;
  size_t _M{0};
  size_t _nbits{0};
  size_t _code_size{0};
  hnswlib::DISTFUNC<dist_t> _dist_func{nullptr};
  void* _dist_func_param{nullptr};
  AlignedTable<dist_t> _dist_table;
  const IndexPQ* _pq{nullptr};
  const hnswlib::HierarchicalNSW<float>* _hnsw{nullptr};
  const dist_t* _query{nullptr};

  // Profile metrics
  Atomic<int64_t> _num_distance_computation;
  Atomic<int64_t> _num_pq_distance_computation;
  Atomic<int64_t> _num_lowerbound_computation;

  // Search parameters
  float gamma{0.8};

  ~UnityOp() override = default;

  explicit UnityOp(const UnityHNSW* uhnsw) {
    U_ASSERT(uhnsw != nullptr);
    U_ASSERT(uhnsw->owned_index_hnsw != nullptr);
    U_ASSERT(uhnsw->owned_index_pq != nullptr);
    U_ASSERT(uhnsw->owned_space != nullptr);
    _pq = uhnsw->owned_index_pq.get();
    _M = _pq->quantizer.M;
    _nbits = _pq->quantizer.nbits;
    _code_size = _pq->code_size;
    _codes = _pq->codes.data();
    _recons_errors = _pq->recons_errors.data();

    _hnsw = uhnsw->owned_index_hnsw.get();
    _dist_func = _hnsw->fstdistfunc_;
    _dist_func_param = _hnsw->dist_func_param_;
  }

  void set_query(const dist_t* query_data) override {
    _query = query_data;
    _dist_table.resize(_pq->quantizer.M * _pq->quantizer.ksub);
    _pq->quantizer.compute_distance_table(_query, _dist_table.data());
    _dist_table_data = _dist_table.data();
  }

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
    assert(_query != nullptr);
    return _dist_func(_query, _hnsw->getDataByInternalId(i), _dist_func_param);
  }

  dist_t relaxed_lowerbound(idx_t i) const override {
    assert(_query != nullptr);
    dist_t a = std::sqrt(estimate(i));
    dist_t b = _recons_errors[i];
    return (a - b) * (a - b) + 2 * gamma * a * b;
  }

  void relaxed_lowerbound8(const Id8& ids, Dist8& dists) const override {
    assert(_query != nullptr);
    _relaxed_lowerbound8(ids, dists);
  }

  dist_t estimate(idx_t i) const override {
    return faiss::distance_single_code<PQDecoderType>(_M, _nbits, _dist_table_data,
                                                      _codes + i * _code_size);
  }

  void set(const std::string& key, const Object& value) override {
    if (key == "gamma") {
      U_THROW_IF_NOT_MSG(value.type == ObjectType::DOUBLE_TYPE,
                         "parameter `ef` must be a double value");
      gamma = static_cast<float>(value.get_double());
    } else {
      U_THROW_FMT("unknown parameter %s", key.c_str());
    }
  }

  void prefetch(idx_t i) const override { prefetch_L1(_codes + _code_size * i); }

  void _prefetch_vector(idx_t i) const { prefetch_L1(_hnsw->getDataByInternalId(i)); }

#ifdef USE_AVX
  void _relaxed_lowerbound8(const Id8& ids, Dist8& dists) const {
    // PQ distances
    float pq_dist[8] = {
        0,
    };

    prefetch_L1(_codes + ids[4] * _code_size);
    prefetch_L1(_codes + ids[5] * _code_size);
    prefetch_L1(_codes + ids[6] * _code_size);
    prefetch_L1(_codes + ids[7] * _code_size);

    faiss::distance_four_codes<PQDecoderType>(
        _M, _nbits, _dist_table_data,                                //
        _codes + ids[0] * _code_size, _codes + ids[1] * _code_size,  //
        _codes + ids[2] * _code_size, _codes + ids[3] * _code_size,  //
        pq_dist[0], pq_dist[1], pq_dist[2], pq_dist[3]);

    // Prefetch reconstruciton errors
    prefetch_L1(_recons_errors + ids[0]);
    prefetch_L1(_recons_errors + ids[1]);
    prefetch_L1(_recons_errors + ids[2]);
    prefetch_L1(_recons_errors + ids[3]);
    prefetch_L1(_recons_errors + ids[4]);
    prefetch_L1(_recons_errors + ids[5]);
    prefetch_L1(_recons_errors + ids[6]);
    prefetch_L1(_recons_errors + ids[7]);

    faiss::distance_four_codes<PQDecoderType>(
        _M, _nbits, _dist_table_data,                                //
        _codes + ids[4] * _code_size, _codes + ids[5] * _code_size,  //
        _codes + ids[6] * _code_size, _codes + ids[7] * _code_size,  //
        pq_dist[4], pq_dist[5], pq_dist[6], pq_dist[7]);

    // PQ distances
    __m256 vec_pq_dist = _mm256_loadu_ps(pq_dist);
    __m256 vec_recons_error = _mm256_set_ps(_recons_errors[ids[7]], _recons_errors[ids[6]],  //
                                            _recons_errors[ids[5]], _recons_errors[ids[4]],  //
                                            _recons_errors[ids[3]], _recons_errors[ids[2]],  //
                                            _recons_errors[ids[1]], _recons_errors[ids[0]]);
    // Lowerbounds
    __m256 vec_lowerbounds = _relaxed_lowerbound8_avx2(gamma, vec_pq_dist, vec_recons_error);
    _mm256_storeu_ps(dists.data(), vec_lowerbounds);
  }

  void _dist_comp8(dist_t max_dist, const Id8& ids, Dist8& dists, Bool8& lt_flags) const {
    // PQ distances
    float pq_dist[8] = {
        0,
    };

    prefetch_L1(_codes + ids[4] * _code_size);
    prefetch_L1(_codes + ids[5] * _code_size);
    prefetch_L1(_codes + ids[6] * _code_size);
    prefetch_L1(_codes + ids[7] * _code_size);

    faiss::distance_four_codes<PQDecoderType>(
        _M, _nbits, _dist_table_data,                                //
        _codes + ids[0] * _code_size, _codes + ids[1] * _code_size,  //
        _codes + ids[2] * _code_size, _codes + ids[3] * _code_size,  //
        pq_dist[0], pq_dist[1], pq_dist[2], pq_dist[3]);

    // Prefetch reconstruciton errors
    prefetch_L1(_recons_errors + ids[0]);
    prefetch_L1(_recons_errors + ids[1]);
    prefetch_L1(_recons_errors + ids[2]);
    prefetch_L1(_recons_errors + ids[3]);
    prefetch_L1(_recons_errors + ids[4]);
    prefetch_L1(_recons_errors + ids[5]);
    prefetch_L1(_recons_errors + ids[6]);
    prefetch_L1(_recons_errors + ids[7]);

    faiss::distance_four_codes<PQDecoderType>(
        _M, _nbits, _dist_table_data,                                //
        _codes + ids[4] * _code_size, _codes + ids[5] * _code_size,  //
        _codes + ids[6] * _code_size, _codes + ids[7] * _code_size,  //
        pq_dist[4], pq_dist[5], pq_dist[6], pq_dist[7]);

    // PQ distances
    __m256 vec_pq_dist = _mm256_loadu_ps(pq_dist);
    __m256 vec_recons_error = _mm256_set_ps(_recons_errors[ids[7]], _recons_errors[ids[6]],  //
                                            _recons_errors[ids[5]], _recons_errors[ids[4]],  //
                                            _recons_errors[ids[3]], _recons_errors[ids[2]],  //
                                            _recons_errors[ids[1]], _recons_errors[ids[0]]);
    // Lowerbounds
    __m256 vec_lowerbounds = _relaxed_lowerbound8_avx2(gamma, vec_pq_dist, vec_recons_error);
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