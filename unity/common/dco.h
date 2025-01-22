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

using Id4 = std::array<unsigned, 4>;
using Id8 = std::array<unsigned, 8>;
using Dist4 = std::array<float, 4>;
using Dist8 = std::array<float, 8>;

template <typename idx_t, typename dist_t>
struct IDistanceComparisonOperator {
  virtual ~IDistanceComparisonOperator() = default;

  /**
   * Set the query data.
   * @param data Pointer to query data.
   */
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
  virtual float dist_comp(dist_t max_dist, idx_t i) const = 0;

  virtual bool dist_comp4(dist_t max_dist, const Id4& ids, Dist4& dists) const {
    bool lt_flag = false;
    for (int i = 0; i < 4; i++) {
      dists[i] = dist_comp(max_dist, ids[i]);
      if (dists[i] > 0) {
        lt_flag = true;
      }
    }
    return lt_flag;
  }

  virtual bool dist_comp8(dist_t max_dist, const Id8& ids, Dist8& dists) const {
    bool lt_flag = false;
    for (int i = 0; i < 8; i++) {
      dists[i] = dist_comp(max_dist, ids[i]);
      if (dists[i] > 0) {
        lt_flag = true;
      }
    }
    return lt_flag;
  }

  /**
   * Compute the exact distance from the query point to the data point at index i.
   * @param i The index of the data point.
   * @return The exact distance value.
   */
  virtual dist_t compute(idx_t i) const = 0;

  /**
   * Compute the lower bound of the distance from the query point to the data point at index i.
   * @param i The index of the data point.
   * @return The lower bound distance value.
   */
  virtual dist_t relaxed_lowerbound(idx_t i) const { return compute(i); };

  /**
   * Estimate the distance from the query point to the data point at index i.
   * @param i The index of the data point.
   * @return The estimated distance value.
   */
  virtual dist_t estimate(idx_t i) const { return compute(i); };

  /**
   * Set the DCO parameter with the specified key and value.
   * @param key The unique identifier for the parameter.
   * @param value The value to be assigned to the parameter.
   */
  virtual void set(const std::string& key, const Object& value) {};

  /**
   * Prefetch data for data point at a specified index. This function can be used to optimize
   * performance by loading data into cache.
   * @param idx The index of the data point to prefetch.
   */
  virtual void prefetch(idx_t idx) const {}

  /**
   * Get a profile dictionary containing performance metrics.
   * This function returns a dictionary with relevant metrics.
   * @return A dictionary containing profile information.
   */
  virtual Dict get_profile() const { return {}; }
};

}  // namespace unity
