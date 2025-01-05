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

#include "top/common/common.h"
#include "top/common/top_assert.h"

namespace top {
namespace constants {

inline constexpr const char* TOP_METRIC = "metric";
inline constexpr const char* TOP_METRIC_L2 = "L2";
inline constexpr const char* TOP_METRIC_IP = "IP";

inline constexpr const char* TOP_HNSW = "hnsw";
inline constexpr const char* TOP_IVFPQR = "ivfpqr";

inline constexpr const char* TOP_NUM_THREADS = "num_threads";
inline constexpr const char* TOP_DIM = "dim";
inline constexpr const char* TOP_EF = "ef";
inline constexpr const char* TOP_HNSWLIB_INDEX_PATH = "hnswlib_index_path";
inline constexpr const char* TOP_PQ_INDEX_PATH = "pq_index_path";

inline Metric metric_map(const std::string& name) {
  if (name == TOP_METRIC_L2) {
    return Metric::L2;
  } else if (name == TOP_METRIC_IP) {
    return Metric::IP;
  }
  TOP_THROW_FMT("unknown metric `%s`, please use valid metric %s or %s", name.c_str(), TOP_METRIC_L2,
                TOP_METRIC_IP);
}
}  // namespace constants
}  // namespace top