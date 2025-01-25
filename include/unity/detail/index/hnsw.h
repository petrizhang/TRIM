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

#include "./index_pq.h"
#include "hnswlib/hnswlib.h"

namespace unity {
namespace detail {

using Space = hnswlib::SpaceInterface<float>;
using HnswlibIndex = hnswlib::HierarchicalNSW<float>;

struct UnityHnsw {
  std::unique_ptr<HnswlibIndex> owned_index_hnsw{nullptr};
  UnityIndexPQ unity_index_pq{nullptr};
  bool _reorderd{false};

  UnityHnsw() = default;
  ~UnityHnsw() = default;

  /// @brief Reorders the PQ codes based on the internal ID order of the HNSW index.
  void _reorder_pq_codes() {
    U_ASSERT(owned_index_hnsw != nullptr && unity_index_pq.owned_index_pq != nullptr);

    faiss::IndexPQ* faiss_index_pq = unity_index_pq.owned_index_pq.get();
    std::vector<uint8_t> reordered_codes(faiss_index_pq->codes.size());
    U_THROW_IF_NOT_MSG(owned_index_hnsw->cur_element_count.load() == (size_t)faiss_index_pq->ntotal,
                       "the HNSW and PQ index must have the same number of data points");
    for (size_t i = 0; i < (size_t)faiss_index_pq->ntotal; i++) {
      auto it = owned_index_hnsw->label_lookup_.find(i);
      if (it == owned_index_hnsw->label_lookup_.end()) {
        U_THROW_FMT("cannot find label %zu in hnsw index", i);
      }
      unsigned int internal_id = it->second;
      auto code_size = faiss_index_pq->code_size;
      std::memcpy(reordered_codes.data() + code_size * internal_id,
                  faiss_index_pq->codes.data() + code_size * i, code_size);
    }
    faiss_index_pq->codes = std::move(reordered_codes);
    _reorderd = true;
  }
};

}  // namespace detail
}  // namespace unity