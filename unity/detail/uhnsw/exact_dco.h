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

  virtual void set_query(const dist_t* query_data) override final { this->_query = query_data; }

  virtual bool distance_less_than(dist_t max_dist, idx_t i, float* dist) const override final {
    assert(_query != nullptr);
    if constexpr (enable_profile) {
      _num_distance_computation.value.fetch_add(1);
    }
    dist_t cur_dist = _dist_func(_query, _hnsw->getDataByInternalId(i), _dist_func_param);
    *dist = cur_dist;
    return cur_dist < max_dist;
  }

  virtual Dict get_profile() const override final {
    Dict dict;
    dict.put("num_distance_computation", Object(_num_distance_computation.value.load()));
    return dict;
  }
};

}  // namespace detail
}  // namespace unity
