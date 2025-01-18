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

#include "unity/common/common.h"
#include "unity/common/u_assert.h"

namespace unity {
namespace constants {

inline constexpr const char* U_METRIC = "metric";
inline constexpr const char* U_METRIC_L2 = "L2";
inline constexpr const char* U_METRIC_IP = "IP";

inline constexpr const char* U_HNSW = "hnsw";
inline constexpr const char* U_IVFPQR = "ivfpqr";

inline constexpr const char* U_NUM_THREADS = "num_threads";
inline constexpr const char* U_DIM = "dim";
inline constexpr const char* U_EF = "ef";
inline constexpr const char* U_HNSWLIB_INDEX_PATH = "hnswlib_index_path";
inline constexpr const char* U_PQ_INDEX_PATH = "pq_index_path";

inline Metric metric_map(const std::string& name) {
  if (name == U_METRIC_L2) {
    return Metric::L2;
  } else if (name == U_METRIC_IP) {
    return Metric::IP;
  }
  U_THROW_FMT("unknown metric `%s`, please use valid metric %s or %s", name.c_str(), U_METRIC_L2,
                U_METRIC_IP);
}
}  // namespace constants
}  // namespace unity