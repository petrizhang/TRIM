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

struct bool4 {
  int32_t mask{0};

  void reset() { mask = 0; }

  bool get0() { return (mask & 0x1) != 0; }

  bool get1() { return (mask & 0x2) != 0; }

  bool get2() { return (mask & 0x4) != 0; }

  bool get3() { return (mask & 0x8) != 0; }

  void set0(bool value) {
    if (value) {
      mask |= 0x1;
    } else {
      mask &= ~0x1;
    }
  }

  void set1(bool value) {
    if (value) {
      mask |= 0x2;
    } else {
      mask &= ~0x2;
    }
  }

  void set2(bool value) {
    if (value) {
      mask |= 0x4;
    } else {
      mask &= ~0x4;
    }
  }

  void set3(bool value) {
    if (value) {
      mask |= 0x8;
    } else {
      mask &= ~0x8;
    }
  }

  bool has_true() { return mask != 0; }

  bool get(int i) {
    assert(i >= 0 && i < 4);
    switch (i) {
      case 0:
        return get0();
      case 1:
        return get1();
      case 2:
        return get2();
      case 3:
        return get3();
      default:
        U_THROW_UNEXPECTED_CODE_PATH;
    }
  }

  void set(int i, bool value) {
    assert(i >= 0 && i < 4);
    switch (i) {
      case 0:
        set0(value);
        break;
      case 1:
        set1(value);
        break;
      case 2:
        set2(value);
        break;
      case 3:
        set3(value);
        break;
      default:
        U_THROW_UNEXPECTED_CODE_PATH;
    }
  }
};

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
  virtual bool distance_less_than(dist_t max_dist, idx_t i, float* dist) const = 0;

  /**
   * Test whether the distances between the query point and the data points at indices i0, i1, i2,
   * i3 are all less than max_dist.
   * @param max_dist The maximum distance threshold.
   * @param i0 The index of the first data point.
   * @param i1 The index of the second data point.
   * @param i2 The index of the third data point.
   * @param i3 The index of the fourth data point.
   * @param result A pointer to store the comparison result.
   * @return Returns true if all four distances are less than max_dist, otherwise returns false.
   */
  virtual bool distance4_less_than(dist_t max_dist, idx_t i0, idx_t i1, idx_t i2, idx_t i3,
                                   float* __restrict dist4, bool4& flag4) const {
    flag4.set0(distance_less_than(max_dist, i0, dist4));
    flag4.set1(distance_less_than(max_dist, i1, dist4 + 1));
    flag4.set2(distance_less_than(max_dist, i2, dist4 + 2));
    flag4.set3(distance_less_than(max_dist, i3, dist4 + 3));
    return flag4.has_true();
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
