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
#include <cstdint>

#include "unity/common/object.h"

namespace unity {

template <typename dist_type, typename ChildClass>
struct DistanceEstimateOperator {
  static constexpr const bool enable_profile = ChildClass::enable_profile;

  std::atomic<int64_t> num_distance_estimation = 0;
  std::atomic<int64_t> num_distance_computation = 0;

  void set_query(const dist_type* data) { ChildClass::set_query_impl(data); }

  void prefetch(int i) { ChildClass::prefetch_impl(i); }

  // Estimate the distance between the query and the i-th data point
  dist_type estimate(int i) { return ChildClass::estimate_impl(i); };
  // Compute the distance between the query and the i-th data point
  dist_type compute(int i) { return ChildClass::compute_impl(i); }

  dist_type estimate_with_profile(int i) {
    num_distance_estimation.fetch_add(1);
    return ChildClass::estimate_impl(i);
  };

  dist_type compute_with_profile(int i) {
    num_distance_computation.fetch_add(1);
    return ChildClass::compute_impl(i);
  }
};

}  // namespace unity
