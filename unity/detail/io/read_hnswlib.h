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

#include <memory>

#include "unity/common/top_assert.h"
#include "unity/detail/hnswlib/hnswlib.h"

namespace unity {
namespace detail {

std::unique_ptr<hnswlib::HierarchicalNSW<float>> read_hnswlib(hnswlib::SpaceInterface<float>* space,
                                                              Metric metric,
                                                              const std::string& path, int dim) {
  U_THROW_IF_NOT_MSG(metric == Metric::L2, "only L2 metric is supported now");
  auto hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space, path, false, 0, false);
  return hnsw;
}

}  // namespace detail
}  // namespace unity
