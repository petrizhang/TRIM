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

#include "top/detail/core/common.h"
#include "top/detail/searcher/hnsw_searcher.h"

namespace top {

inline std::unique_ptr<HNSWSearcher> create_hnsw_searcher(const Graph<int>& graph,
                                                          const std::string& metric) {
  FAISS_THROW_IF_MSG(metric != "L2", "only L2 metric is supported now");
  return std::make_unique<HNSWSearcher>(graph, FP32Quantizer());
}

}  // namespace top