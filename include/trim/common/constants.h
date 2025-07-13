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

#include "trim/common/common.h"
#include "trim/common/t_assert.h"

namespace trim {
namespace constants {

inline constexpr const char* T_METRIC = "metric";
inline constexpr const char* T_METRIC_L2 = "L2";
inline constexpr const char* T_METRIC_IP = "IP";

inline constexpr const char* T_HNSW = "hnsw";
inline constexpr const char* T_IVFPQ = "ivfpq";

inline constexpr const char* T_DCO = "dco";
inline constexpr const char* T_DCO_EXACT = "exact";
inline constexpr const char* T_DCO_TRIM = "trim";

inline constexpr const char* T_ENABLE_PROFILE = "enable_profile";
inline constexpr const char* T_NUM_THREADS = "num_threads";
inline constexpr const char* T_DIM = "dim";
inline constexpr const char* T_EF = "ef";
inline constexpr const char* T_HNSWLIB_INDEX_PATH = "hnswlib_index_path";
inline constexpr const char* T_IVFPQ_INDEX_PATH = "ivfpq_index_path";
inline constexpr const char* DATA_PATH = "data_path";
inline constexpr const char* T_PQ_INDEX_PATH = "pq_index_path";
inline constexpr const char* T_USE_OPQ = "use_opq";
inline constexpr const char* RANDOM_LANDMARK_SIZE = "random_landmark_size";

inline Metric metric_map(const std::string& name) {
  if (name == T_METRIC_L2) {
    return Metric::L2;
  } else if (name == T_METRIC_IP) {
    return Metric::IP;
  }
  T_THROW_FMT("unknown metric `%s`, please use valid metric %s or %s", name.c_str(), T_METRIC_L2,
              T_METRIC_IP);
}
}  // namespace constants
}  // namespace trim