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

#include "trim/common/atomic.h"
#include "trim/common/dco.h"
#include "trim/common/prefetch.h"
#include "trim/detail/index/hnsw.h"

namespace trim {
namespace detail {

template <bool enable_profile = false>
struct ExactDCO final : IDistanceComparisonOperator<unsigned, float> {
  using Parent = IDistanceComparisonOperator<unsigned, float>;
  using idx_t = unsigned;
  using dist_t = float;

  const dist_t* _query{nullptr};
  hnswlib::DISTFUNC<dist_t> _dist_func{nullptr};
  void* _dist_func_param{nullptr};
  const HnswlibIndex* _hnsw{nullptr};
  mutable Atomic<int64_t> _num_distance_computation{0};

  ExactDCO() = default;

  ~ExactDCO() override = default;

  explicit ExactDCO(const TrimHNSW* uhnsw) {
    T_ASSERT(uhnsw != nullptr);
    _hnsw = uhnsw->owned_index_hnsw.get();
    _dist_func = _hnsw->fstdistfunc_;
    _dist_func_param = _hnsw->dist_func_param_;
  }

  explicit ExactDCO(const HnswlibIndex* hnsw) {
    T_ASSERT(hnsw != nullptr);
    _hnsw = hnsw;
    _dist_func = _hnsw->fstdistfunc_;
    _dist_func_param = _hnsw->dist_func_param_;
  }

  void set_query(const dist_t* query_data) override { this->_query = query_data; }

  bool dist_comp(dist_t max_dist, idx_t i, float& dist) const override {
    assert(_query != nullptr);
    if constexpr (enable_profile) {
      _num_distance_computation.value.fetch_add(1);
    }
    dist = _dist_func(_query, _hnsw->getDataByInternalId(i), _dist_func_param);
    return dist < max_dist;
  }

  dist_t compute(idx_t i) const override {
    assert(_query != nullptr);
    if constexpr (enable_profile) {
      _num_distance_computation.value.fetch_add(1);
    }
    return _dist_func(_query, _hnsw->getDataByInternalId(i), _dist_func_param);
  }

  std::unique_ptr<IDCO> clone() const override {
    auto cloned = std::make_unique<ExactDCO>();
    *cloned = *this;
    return cloned;
  }

  void prefetch(idx_t i) const override { prefetch_l1(_hnsw->getDataByInternalId(i)); }

  float get_gamma() { return 0.0; }

  Dict get_profile() const override {
    Dict dict;
    dict.put("num_distance_computation", Object(_num_distance_computation.value.load()));
    return dict;
  }
};

}  // namespace detail
}  // namespace trim

