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

#include <atomic>
#include <cassert>

#include "unity/common/atomic.h"
#include "unity/common/dco.h"
#include "unity/detail/uhnsw/unity_hnsw.h"

namespace unity {
namespace detail {

template <bool enable_profile = false>
struct ExactDCO final : IDistanceComparisonOperator<unsigned, float> {
  using idx_t = unsigned;
  using dist_t = float;

  const dist_t* _query{nullptr};
  hnswlib::DISTFUNC<dist_t> _dist_func{nullptr};
  void* _dist_func_param{nullptr};
  const HnswlibIndex* _hnsw{nullptr};
  mutable Atomic<int64_t> _num_distance_computation{0};

  explicit ExactDCO(const UnityHNSW* uhnsw) {
    U_ASSERT(uhnsw != nullptr);
    _hnsw = uhnsw->owned_index_hnsw.get();
    _dist_func = _hnsw->fstdistfunc_;
    _dist_func_param = _hnsw->dist_func_param_;
  }

  void set_query(const dist_t* query_data) override final { this->_query = query_data; }

  bool distance_less_than(dist_t max_dist, idx_t i, float* dist) const override final {
    assert(_query != nullptr);
    if constexpr (enable_profile) {
      _num_distance_computation.value.fetch_add(1);
    }
    dist_t cur_dist = _dist_func(_query, _hnsw->getDataByInternalId(i), _dist_func_param);
    *dist = cur_dist;
    return cur_dist < max_dist;
  }

  bool distance4_less_than(dist_t max_dist, idx_t i0, idx_t i1, idx_t i2, idx_t i3,
                           float* __restrict dist4, bool4& flag4) const override final {
    assert(_query != nullptr);
    if constexpr (enable_profile) {
      _num_distance_computation.value.fetch_add(4);
    }
    return _distance4_less_than_simd(max_dist, i0, i1, i2, i3, dist4, flag4);
  }

  dist_t compute(idx_t i) const override final {
    assert(_query != nullptr);
    return _dist_func(_query, _hnsw->getDataByInternalId(i), _dist_func_param);
  }

  bool _distance4_less_than_simd(dist_t max_dist, idx_t i0, idx_t i1, idx_t i2, idx_t i3,
                                 float* __restrict dist4, bool4& flag4) const {
#ifdef USE_SSE
    prefetch(i1);
    dist4[0] = _dist_func(_query, _hnsw->getDataByInternalId(i0), _dist_func_param);

    prefetch(i2);
    dist4[1] = _dist_func(_query, _hnsw->getDataByInternalId(i1), _dist_func_param);

    prefetch(i3);
    dist4[2] = _dist_func(_query, _hnsw->getDataByInternalId(i2), _dist_func_param);

    dist4[3] = _dist_func(_query, _hnsw->getDataByInternalId(i3), _dist_func_param);

    __m128 dist_vec = _mm_loadu_ps(dist4);
    __m128 max_dist_vec = _mm_set1_ps(max_dist);
    __m128 less_than_result = _mm_cmplt_ps(dist_vec, max_dist_vec);
    flag4.mask = _mm_movemask_ps(less_than_result);
    return flag4.has_true();
#else
    return _distance4_less_than_plain(max_dist, i0, i1, i2, i3, dist4, flag4);
#endif
  }

  bool _distance4_less_than_plain(dist_t max_dist, idx_t i0, idx_t i1, idx_t i2, idx_t i3,
                                  float* __restrict dist4, bool4& flag4) const {
    prefetch(i1);
    flag4.set_bool0(distance_less_than(max_dist, i0, dist4));

    prefetch(i2);
    flag4.set_bool1(distance_less_than(max_dist, i1, dist4 + 1));

    prefetch(i3);
    flag4.set_bool2(distance_less_than(max_dist, i2, dist4 + 2));

    flag4.set_bool3(distance_less_than(max_dist, i3, dist4 + 3));
    return flag4.has_true();
  }

  void prefetch(idx_t i) const override final {
#ifdef USE_SSE
    _mm_prefetch(_hnsw->getDataByInternalId(i), _MM_HINT_T0);
#endif
  }

  Dict get_profile() const override final {
    Dict dict;
    dict.put("num_distance_computation", Object(_num_distance_computation.value.load()));
    return dict;
  }
};

}  // namespace detail
}  // namespace unity
