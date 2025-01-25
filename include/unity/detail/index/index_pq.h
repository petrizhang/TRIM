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
#include "unity/common/dco.h"
#include "unity/common/u_assert.h"
#include "unity/util/thread_pool.h"

namespace unity {
namespace detail {

struct UnityIndexPq {
  std::unique_ptr<faiss::IndexPQ> owned_index_pq{nullptr};
  faiss::AlignedTable<uint8_t> codes;
  /// Distances between data points and their PQ centroids.
  faiss::AlignedTable<float> recons_errors;
  bool has_recons_errors{false};

  explicit UnityIndexPq(std::unique_ptr<faiss::IndexPQ> owned_index)
      : owned_index_pq(std::move(owned_index)) {}

  /// @brief Compute the distances between data points and their PQ centroids.
  void compute_pq_reconstruction_errors(IDco* dco, ctpl::thread_pool& pool) {
    U_THROW_IF_NOT_MSG(owned_index_pq != nullptr, "index is nullptr");

    auto ntotal = owned_index_pq->ntotal;
    auto* index_pq = owned_index_pq.get();

    int batch_size = owned_index_pq->ntotal / pool.size();
    std::vector<std::future<void>> futures;

    int end = owned_index_pq->ntotal;
    recons_errors.resize(ntotal);

    float* out = recons_errors.data();
    for (int task_start = 0, task_end = 0; task_end < end; task_start += batch_size) {
      task_end = task_start + batch_size;
      if (task_end > end) {
        task_end = end;
      }
      auto future = pool.push([=](int task_id) {
        std::vector<float> recons(index_pq->pq.d);
        for (int j = task_start; j < task_end; j++) {
          index_pq->reconstruct(j, recons.data());
          dco->set_query(recons.data());
          float dist = dco->compute(j);
          out[j] = std::sqrt(dist);
        }
      });
      futures.push_back(std::move(future));
    }

    for (auto& f : futures) {
      f.get();
    }
    this->has_recons_errors = true;
  }
};

}  // namespace detail
}  // namespace unity