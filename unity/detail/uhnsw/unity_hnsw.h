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

#include "unity/thirdparty/hnswlib/hnswlib.h"
#include "unity/detail/quantizer/index_pq.h"

namespace unity {
namespace detail {

using Space = hnswlib::SpaceInterface<float>;
using HnswlibIndex = hnswlib::HierarchicalNSW<float>;

struct UnityHNSW {
  std::unique_ptr<Space> owned_space{nullptr};
  std::unique_ptr<HnswlibIndex> owned_index_hnsw{nullptr};
  std::unique_ptr<IndexPQ> owned_index_pq{nullptr};
  bool _reorderd{false};

  UnityHNSW() = default;
  ~UnityHNSW() = default;

  /// @brief Reorders the PQ codes based on the internal ID order of the HNSW index.
  void _reorder_pq_codes();

  /// @brief Compute the distances between data points and their PQ centroids.
  void _compute_pq_reconstruction_errors(ctpl::thread_pool& pool);
};

inline void UnityHNSW::_reorder_pq_codes() {
  assert(owned_index_hnsw != nullptr && owned_index_pq != nullptr);
  faiss::AlignedTable<uint8_t> reordered_codes(owned_index_pq->codes.size());
  U_THROW_IF_NOT_MSG(owned_index_hnsw->cur_element_count.load() == (size_t)owned_index_pq->ntotal,
                     "the HNSW and PQ index must have the same number of data points");
  for (size_t i = 0; i < (size_t)owned_index_pq->ntotal; i++) {
    auto it = owned_index_hnsw->label_lookup_.find(i);
    if (it == owned_index_hnsw->label_lookup_.end()) {
      U_THROW_FMT("cannot find label %zu in hnsw index", i);
    }
    unsigned int internal_id = it->second;
    auto code_size = owned_index_pq->code_size;
    std::memcpy(reordered_codes.data() + code_size * internal_id,
                owned_index_pq->codes.data() + code_size * i, code_size);
  }
  owned_index_pq->codes = reordered_codes;
  _reorderd = true;
}

inline void UnityHNSW::_compute_pq_reconstruction_errors(ctpl::thread_pool& pool) {
  assert(owned_index_pq != nullptr);
  U_THROW_IF_NOT_MSG(_reorderd, "PQ codes have not been reordered");

  auto dim = owned_index_pq->d;
  auto ntotal = owned_index_pq->ntotal;
  auto* index_hnsw = owned_index_hnsw.get();
  auto* index_pq = owned_index_pq.get();

  int batch_size = owned_index_pq->ntotal / pool.size();
  std::vector<std::future<void>> futures;
  hnswlib::L2Space space(dim);
  hnswlib::DISTFUNC<float> dist_func = space.get_dist_func();
  void* dist_func_param = space.get_dist_func_param();

  int end = owned_index_pq->ntotal;
  owned_index_pq->recons_errors.resize(ntotal);

  float* out = owned_index_pq->recons_errors.data();
  for (int task_start = 0, task_end = 0; task_end < end; task_start += batch_size) {
    task_end = task_start + batch_size;
    if (task_end > end) {
      task_end = end;
    }
    auto future = pool.push([=](int task_id) {
      std::vector<float> recons(index_pq->quantizer.d);
      for (int j = task_start; j < task_end; j++) {
        index_pq->reconstruct(j, recons.data());
        float dist = dist_func(index_hnsw->getDataByInternalId(j), recons.data(), dist_func_param);
        out[j] = std::sqrt(dist);
      }
    });
    futures.push_back(std::move(future));
  }

  for (auto& f : futures) {
    f.get();
  }
}

}  // namespace detail
}  // namespace unity