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
#include "trim/common/atomic.h"
#include "trim/common/dco.h"
#include "trim/common/object.h"
#include "trim/common/prefetch.h"
#include "trim/common/setter_proxy.h"
#include "trim/detail/index/hnsw.h"

namespace trim {
namespace detail {

template <typename PQDecoderType, bool enable_profile = false>
struct TrimDCO final : SetterProxy<TrimDCO<PQDecoderType, enable_profile>>, IDCO {
  using This = TrimDCO<PQDecoderType, enable_profile>;
  using Proxy = SetterProxy<This>;
  using Parent = IDCO;
  using idx_t = unsigned;
  using dist_t = float;

  /* Data structures for query */
  const dist_t* dist_table_data{nullptr};
  const uint8_t* codes;
  const float* recons_errors;
  const dist_t* query{nullptr};
  size_t m{0};
  size_t nbits{0};
  size_t code_size{0};
  size_t _random_landmark_size{0};
  hnswlib::DISTFUNC<dist_t> dist_func{nullptr};
  void* dist_func_param{nullptr};
  faiss::AlignedTable<dist_t> dist_table;

  /* Indexes */
  const faiss::IndexPQ* faiss_index_pq{nullptr};
  const hnswlib::HierarchicalNSW<float>* hnsw{nullptr};

  /* Profile metrics */
  Atomic<int64_t> num_distance_computation;
  Atomic<int64_t> num_pq_distance_computation;
  Atomic<int64_t> num_lowerbound_computation;

  /* For storing random landmarks */
  int* _random_landmarks{nullptr};

  /* Search parameters */
  float gamma{0.8};

  ~TrimDCO() override = default;

  TrimDCO() = default;

  explicit TrimDCO(const TrimHNSW* p_uhnsw, size_t random_landmark_size = 0)
      : SetterProxy<This>("TrimDCO") {
    T_ASSERT(p_uhnsw != nullptr);
    T_ASSERT(p_uhnsw->owned_index_hnsw != nullptr);
    hnsw = p_uhnsw->owned_index_hnsw.get();
    dist_func = hnsw->fstdistfunc_;
    dist_func_param = hnsw->dist_func_param_;

    Proxy::template bind<DOUBLE_TYPE>("gamma", &This::set_gamma);
    Proxy::template bind<INTEGER_TYPE>("random_landmark_size", &This::set_random_landmark_size);

    _random_landmark_size = random_landmark_size;

    if (_random_landmark_size > 0) {
      _random_landmarks = new int[_random_landmark_size];

      int total_data_size = hnsw->cur_element_count;
      std::cout << "total_data_size:" << total_data_size << std::endl;
      std::vector<int> data_ids(total_data_size);
      for (int i = 0; i < total_data_size; ++i) {
        data_ids[i] = i;
      }

      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(data_ids.begin(), data_ids.end(), g);

      for (int i = 0; i < _random_landmark_size; ++i) {
        _random_landmarks[i] = data_ids[i];
      }

      // 输出随机选择的地标 ID
      std::cout << "Random landmarks: ";
      for (int i = 0; i < _random_landmark_size; ++i) {
        std::cout << _random_landmarks[i] << " ";
      }
      std::cout << std::endl;

      float* xl_distances = new float[total_data_size * _random_landmark_size];
      for (int i = 0; i < total_data_size; i++) {
        for (int l = 0; l < _random_landmark_size; l++) {
          if (i == _random_landmarks[l]) {
            xl_distances[i * _random_landmark_size + l] = 0;
          } else {
            xl_distances[i * _random_landmark_size + l] =
                sqrt(dist_func(hnsw->getDataByInternalId(i),
                               hnsw->getDataByInternalId(_random_landmarks[l]), dist_func_param));
          }
        }
      }
      recons_errors = xl_distances;
    } else {
      T_ASSERT(p_uhnsw->trim_index_pq.index_pq != nullptr);
      faiss_index_pq = p_uhnsw->trim_index_pq.index_pq;
      m = faiss_index_pq->pq.M;
      nbits = faiss_index_pq->pq.nbits;
      code_size = faiss_index_pq->code_size;
      codes = p_uhnsw->trim_index_pq.index_pq->codes.data();
      recons_errors = p_uhnsw->trim_index_pq.recons_errors.data();
    }
  }

  void set_query(const dist_t* query_data) override {
    query = query_data;
    if (_random_landmark_size == 0) {
      dist_table.resize(faiss_index_pq->pq.M * faiss_index_pq->pq.ksub);
      faiss_index_pq->pq.compute_distance_table(query, dist_table.data());
    } else {
      dist_table.resize(_random_landmark_size);
      for (int i = 0; i < _random_landmark_size; i++) {
        dist_table[i] = compute(_random_landmarks[i]);
        // dist_table[i] = fvec_L2sqr(query, _random_landmarks[i]);
      }
    }
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
    // std::cout << "id:" << i << std::endl;
    // std::cout << "compute:" << compute(i) << std::endl;
    assert(query != nullptr);
    if (_random_landmark_size > 0) {
      dist_t max_lowerbound = -1.0f;
      for (int l = 0; l < _random_landmark_size; l++) {
        dist_t a = sqrt(dist_table_data[l]);
        dist_t b = recons_errors[i * _random_landmark_size + l];
        dist_t vec_lowerbound = (a - b) * (a - b) + 2 * gamma * a * b;
        max_lowerbound = std::max(max_lowerbound, vec_lowerbound);
      }
      return max_lowerbound;
    } else {
      dist_t a = std::sqrt(estimate(i));
      dist_t b = recons_errors[i];
      return (a - b) * (a - b) + 2 * gamma * a * b;
    }
  }

  void relaxed_lowerbound8(const Id8& ids, Dist8& dists) const override {
    assert(query != nullptr);
    _relaxed_lowerbound8(ids, dists);
  }

  dist_t estimate(idx_t i) const override {
    return faiss::distance_single_code<PQDecoderType>(m, nbits, dist_table_data,
                                                      codes + i * code_size);
  }

  std::unique_ptr<IDCO> clone() const override {
    auto cloned = std::make_unique<TrimDCO>();
    *cloned = *this;
    return cloned;
  }

  void prefetch(idx_t i) const override { prefetch_l1(codes + code_size * i); }

  void set_gamma(float gamma) { this->gamma = gamma; }

  void set_random_landmark_size(size_t size = 0) { _random_landmark_size = size; }

  float get_gamma() { return gamma; }

  size_t get_random_landmark_size() { return _random_landmark_size; }

  void _prefetch_vector(idx_t i) const { prefetch_l1(hnsw->getDataByInternalId(i)); }

#ifdef USE_AVX
  void _relaxed_lowerbound8(const Id8& ids, Dist8& dists) const {
    if (_random_landmark_size > 0) {
      __m256 vec_max_lowerbound = _mm256_set1_ps(-1.0f);

      for (int i = 0; i < _random_landmark_size; i++) {
        // Prefetch dist_table_data
        prefetch_l1(dist_table_data + i);

        // ql distances
        __m256 q_landmark_dist = _mm256_set1_ps(dist_table_data[i]);

        // Prefetch reconstruction errors
        for (int j = 0; j < 8; ++j) {
          prefetch_l1(recons_errors + ids[j] * _random_landmark_size + i);
        }

        // xl distances
        __m256 vec_recons_error = _mm256_set_ps(recons_errors[ids[7] * _random_landmark_size + i],
                                                recons_errors[ids[6] * _random_landmark_size + i],
                                                recons_errors[ids[5] * _random_landmark_size + i],
                                                recons_errors[ids[4] * _random_landmark_size + i],
                                                recons_errors[ids[3] * _random_landmark_size + i],
                                                recons_errors[ids[2] * _random_landmark_size + i],
                                                recons_errors[ids[1] * _random_landmark_size + i],
                                                recons_errors[ids[0] * _random_landmark_size + i]);

        // Lowerbounds
        __m256 vec_lowerbounds =
            _relaxed_lowerbound8_avx2(gamma, q_landmark_dist, vec_recons_error);

        // Compare and update max_lowerbound
        vec_max_lowerbound = _mm256_max_ps(vec_max_lowerbound, vec_lowerbounds);
      }

      // Store the final max_lowerbound values to dists
      _mm256_storeu_ps(dists.data(), vec_max_lowerbound);

      return;
    }

    // PQ distances
    float pq_dist[8] = {
        0,
    };

    prefetch_l1(codes + ids[4] * code_size);
    prefetch_l1(codes + ids[5] * code_size);
    prefetch_l1(codes + ids[6] * code_size);
    prefetch_l1(codes + ids[7] * code_size);

    faiss::distance_four_codes<PQDecoderType>(
        m, nbits, dist_table_data,                               //
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
        m, nbits, dist_table_data,                               //
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
    __m256 vec_lowerbounds = _relaxed_lowerbound8_avx2(gamma, vec_pq_dist, vec_recons_error);
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
        m, nbits, dist_table_data,                               //
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
        m, nbits, dist_table_data,                               //
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
    if (_random_landmark_size > 0) {
      float max_lowerbound[8] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

      for (int i = 0; i < _random_landmark_size; i++) {
        prefetch_l1(dist_table_data + i);
        float q_landmark_dist = sqrt(dist_table_data[i]);

        for (int j = 0; j < 8; ++j) {
          prefetch_l1(recons_errors + ids[j] * _random_landmark_size + i);
          float vec_recons_error = recons_errors[ids[j] * _random_landmark_size + i];
          float vec_lowerbound =
              (q_landmark_dist - vec_recons_error) * (q_landmark_dist - vec_recons_error) +
              2 * gamma * q_landmark_dist * vec_recons_error;
          max_lowerbound[j] = std::max(max_lowerbound[j], vec_lowerbound)
        }
      }

      for (int i = 0; i < 8; ++i) {
        dists[i] = max_lowerbound[i];
      }

      return;
    } else {
      Parent::relaxed_lowerbound8(ids, dists);
    }
  }

  void _dist_comp8(dist_t max_dist, const Id8& ids, Dist8& dists) const {
    Parent::dist_comp8(max_dist, ids, dists);
  }

#endif
};

template <bool enable_profile = false>
using TrimDCO8 = TrimDCO<faiss::PQDecoder8, enable_profile>;

template <typename T>
constexpr const bool is_trim_dco_v = false;

template <typename T, bool v>
constexpr const bool is_trim_dco_v<TrimDCO<T, v>> = true;

}  // namespace detail
}  // namespace trim