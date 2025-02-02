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
#include "faiss/IndexPreTransform.h"
#include "faiss/utils/AlignedTable.h"
#include "unity/common/dco.h"
#include "unity/common/u_assert.h"
#include "unity/detail/index/pq.h"
#include "unity/util/thread_pool.h"

namespace unity {
namespace detail {

struct UnityIndexOPQ {
  UnityIndexPQ pq{nullptr, nullptr};
  faiss::IndexPreTransform* transform{nullptr};

  UnityIndexOPQ(std::unique_ptr<faiss::Index> owned_index, faiss::IndexPQ* index_pq,
                faiss::IndexPreTransform* transform)
      : pq(std::move(owned_index), index_pq), transform(transform) {}

  /// Compute distances between data points and their PQ centroids.
  void compute_pq_reconstruction_errors(IDCO* dco, ctpl::thread_pool& pool) {
    U_THROW_IF_NOT(pq.owned_index != nullptr && pq.index_pq != nullptr);

    size_t ntotal = pq.index_pq->ntotal;
    int batch_size = pq.index_pq->ntotal / pool.size();
    std::vector<std::future<void>> futures;
    faiss::IndexPQ* faiss_index_pq = pq.index_pq;
    faiss::IndexPreTransform* faiss_trans = transform;

    int end = pq.index_pq->ntotal;
    pq.recons_errors.resize(ntotal);

    float* out = pq.recons_errors.data();
    for (int task_start = 0, task_end = 0; task_end < end; task_start += batch_size) {
      task_end = task_start + batch_size;
      if (task_end > end) {
        task_end = end;
      }
      auto future =
          pool.push([faiss_trans, faiss_index_pq, dco, task_start, task_end, out](int task_id) {
            auto dco_copy = dco->clone();
            std::vector<float> recons(faiss_index_pq->pq.d);
            for (int j = task_start; j < task_end; j++) {
              if (faiss_trans != nullptr) {
                faiss_trans->reconstruct(j, recons.data());
              } else {
                faiss_index_pq->reconstruct(j, recons.data());
              }
              dco_copy->set_query(recons.data());
              float dist = dco_copy->compute(j);
              out[j] = std::sqrt(dist);
            }
          });
      futures.push_back(std::move(future));
    }

    for (auto& f : futures) {
      f.get();
    }

    pq.has_recons_errors = true;
  }
};

}  // namespace detail
}  // namespace unity