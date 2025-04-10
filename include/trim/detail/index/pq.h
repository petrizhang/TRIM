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

#include "faiss/IndexPQ.h"
#include "faiss/utils/AlignedTable.h"
#include "trim/common/dco.h"
#include "trim/common/t_assert.h"
#include "trim/util/thread_pool.h"

namespace trim {
namespace detail {

struct TrimIndexPQ {
  std::unique_ptr<faiss::Index> owned_index{nullptr};
  faiss::IndexPQ* index_pq{nullptr};
  /// Distances between data points and their PQ centroids.
  faiss::AlignedTable<float> recons_errors;
  bool has_recons_errors{false};

  explicit TrimIndexPQ(std::unique_ptr<faiss::Index> owned_index, faiss::IndexPQ* index_pq)
      : owned_index(std::move(owned_index)), index_pq(index_pq) {}

};

}  // namespace detail
}  // namespace trim