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

#include <string>

#include "trim/common/dco.h"
#include "trim/common/object.h"

namespace trim {

struct ISearcher {
  virtual ~ISearcher() = default;
  virtual void set_data(float* data) {};
  virtual void set(const std::string& key, const Object& value) {}
  virtual void try_set(const std::string& key, const Object& value) {}
  virtual const float* get_data(unsigned i) const { T_THROW_NOT_IMPLEMENTED; }
  virtual size_t num_data_points() const { T_THROW_NOT_IMPLEMENTED; }
  virtual void ann_search(const float* q, int k, int* dst) const = 0;
  virtual void range_search(const float* q, float radius, std::vector<int> &result) const = 0;
  virtual void optimize(int num_threads) = 0;
  virtual IDCO* get_dco() const = 0;
  virtual Dict get_profile() const = 0;
  virtual void clear_pruning_ratio() const = 0;
  virtual float get_pruning_ratio() const = 0;
};

}  // namespace trim
