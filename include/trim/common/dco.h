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

#include "trim/common/object.h"

namespace trim {

struct Bool8 {
  int32_t mask{0};

  explicit Bool8(bool value) : mask(value ? 0xFF : 0) {}

  explicit Bool8(int32_t value) : mask(value) {}

  void reset_true() { mask = 0; }

  void reset_false() { mask = 0xFF; }

  bool get0() const { return (mask & (1 << 0)) != 0; }
  bool get1() const { return (mask & (1 << 1)) != 0; }
  bool get2() const { return (mask & (1 << 2)) != 0; }
  bool get3() const { return (mask & (1 << 3)) != 0; }
  bool get4() const { return (mask & (1 << 4)) != 0; }
  bool get5() const { return (mask & (1 << 5)) != 0; }
  bool get6() const { return (mask & (1 << 6)) != 0; }
  bool get7() const { return (mask & (1 << 7)) != 0; }

  void set0(bool value) { mask = value ? (mask | (1 << 0)) : (mask & ~(1 << 0)); }
  void set1(bool value) { mask = value ? (mask | (1 << 1)) : (mask & ~(1 << 1)); }
  void set2(bool value) { mask = value ? (mask | (1 << 2)) : (mask & ~(1 << 2)); }
  void set3(bool value) { mask = value ? (mask | (1 << 3)) : (mask & ~(1 << 3)); }
  void set4(bool value) { mask = value ? (mask | (1 << 4)) : (mask & ~(1 << 4)); }
  void set5(bool value) { mask = value ? (mask | (1 << 5)) : (mask & ~(1 << 5)); }
  void set6(bool value) { mask = value ? (mask | (1 << 6)) : (mask & ~(1 << 6)); }
  void set7(bool value) { mask = value ? (mask | (1 << 7)) : (mask & ~(1 << 7)); }

  bool has_true() const { return mask != 0; }

  // General getter for any index (0-7)
  bool get(int i) const {
    assert(i >= 0 && i < 8);  // Ensure index is within bounds
    switch (i) {
      case 0:
        return get0();
      case 1:
        return get1();
      case 2:
        return get2();
      case 3:
        return get3();
      case 4:
        return get4();
      case 5:
        return get5();
      case 6:
        return get6();
      default:
        return get7();
    }
  }

  // General setter for any index (0-7)
  void set(int i, bool value) {
    assert(i >= 0 && i < 8);  // Ensure index is within bounds
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
      case 4:
        set4(value);
        break;
      case 5:
        set5(value);
        break;
      case 6:
        set6(value);
        break;
      default:
        set7(value);
        break;
    }
  }
};

using Id8 = std::array<unsigned, 8>;
using Dist8 = std::array<float, 8>;

template <typename idx_t, typename dist_t>
struct IDistanceComparisonOperator {
  virtual ~IDistanceComparisonOperator() = default;

  /**
   * Set the query data.
   * @param data Pointer to query data.
   */
  virtual void set_query(const dist_t* data) = 0;

  virtual void set(const std::string& key, const Object& value) {}

  virtual void try_set(const std::string& key, const Object& value) {}

  /**
   * Test whether the distance between the query point and the data point at index i is less than
   * max_dist. Checks if the distance from the query point to the data point at index i is less than
   * the specified maximum distance.
   * @param max_dist The maximum distance threshold.
   * @param i The index of the data point.
   * @param dist A reference to store the calculated distance. If the distance is less than
   * max_dist, the value pointed to by this pointer will be updated to the actual distance.
   * @return Returns true if the distance is less than max_dist, otherwise returns false.
   */
  virtual bool dist_comp(dist_t max_dist, idx_t i, float& dist) const = 0;

  virtual void dist_comp8(dist_t max_dist, const Id8& ids, Dist8& dists, Bool8& lt_flags) const {
    for (int i = 0; i < 8; i++) {
      lt_flags.set(i, dist_comp(max_dist, ids[i], dists[i]));
    }
  }

  /**
   * Compute the exact distance from the query point to the data point at index i.
   * @param i The index of the data point.
   * @return The exact distance value.
   */
  virtual dist_t compute(idx_t i) const = 0;

  virtual void compute8(const Id8& ids, Dist8& dists) const {
    for (int i = 0; i < 8; i++) {
      dists[i] = compute(ids[i]);
    }
  }

  /**
   * Compute the lower bound of the distance from the query point to the data point at index i.
   * @param i The index of the data point.
   * @return The lower bound distance value.
   */
  virtual dist_t relaxed_lowerbound(idx_t i) const { return compute(i); };

  virtual void relaxed_lowerbound8(const Id8& ids, Dist8& dists) const { compute8(ids, dists); }

  /**
   * Estimate the distance from the query point to the data point at index i.
   * @param i The index of the data point.
   * @return The estimated distance value.
   */
  virtual dist_t estimate(idx_t i) const { return compute(i); };

  virtual void estimate8(const Id8& ids, Dist8& dists) const { compute8(ids, dists); }

  virtual std::unique_ptr<IDistanceComparisonOperator<idx_t, dist_t>> clone() const = 0;

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

  virtual float get_gamma() const { return 0.0; }

  virtual size_t get_random_landmark_size() const { return 0; }
};

using IDCO = IDistanceComparisonOperator<unsigned, float>;

}  // namespace trim
