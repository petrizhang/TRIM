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

template <typename dist_t>
struct ComparisonResult4 {
  dist_t d0 = -1;
  dist_t d1 = -1;
  dist_t d2 = -1;
  dist_t d3 = -1;
  bool is_d0_valid = false;
  bool is_d1_valid = false;
  bool is_d2_valid = false;
  bool is_d3_valid = false;
};

template <typename idx_t, typename dist_t>
struct IDistanceComparisonOperator {
  virtual ~IDistanceComparisonOperator() = default;

  virtual void set_query(const dist_t* data) = 0;

  /**
   * Test whether the distance between the query point and the data point at index i is less than
   * max_dist. Checks if the distance from the query point to the data point at index i is less than
   * the specified maximum distance.
   * @param max_dist The maximum distance threshold.
   * @param i The index of the data point.
   * @param dist A pointer to store the calculated distance. If the distance is less than max_dist,
   * the value pointed to by this pointer will be updated to the actual distance.
   * @return Returns true if the distance is less than max_dist, otherwise returns false.
   */
  virtual bool distance_less_than(dist_t max_dist, idx_t i, float* dist) const = 0;

  /**
   * Test whether the distances between the query point and the data points at indices i0, i1, i2,
   * i3 are all less than max_dist. Checks if the distances from the query point to the data points
   * at indices i0, i1, i2, i3 are all less than the specified maximum distance.
   * @param max_dist The maximum distance threshold.
   * @param i0 The index of the first data point.
   * @param i1 The index of the second data point.
   * @param i2 The index of the third data point.
   * @param i3 The index of the fourth data point.
   * @param result A pointer to store the comparison result.
   * @return Returns true if all four distances are less than max_dist, otherwise returns false.
   */
  virtual bool distance4_less_than(dist_t max_dist, idx_t i0, idx_t i1, idx_t i2, idx_t i3,
                                   ComparisonResult4<dist_t>* __restrict result) const {
    result->is_d0_valid = distance_less_than(max_dist, i0, &result->d0);
    result->is_d1_valid = distance_less_than(max_dist, i1, &result->d1);
    result->is_d2_valid = distance_less_than(max_dist, i2, &result->d2);
    result->is_d3_valid = distance_less_than(max_dist, i3, &result->d3);
    return result->is_d0_valid && result->is_d1_valid && result->is_d2_valid && result->is_d3_valid;
  }

  virtual void prefetch(idx_t) {}

  virtual Dict get_profile() const { return {}; }
};

}  // namespace unity
