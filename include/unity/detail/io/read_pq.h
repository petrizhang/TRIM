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

#include "faiss/Index.h"
#include "unity/common/u_assert.h"
#include "unity/detail/io/read_faiss.h"

namespace unity {
namespace detail {

template <typename T>
inline std::unique_ptr<T> dyn_cast(std::unique_ptr<faiss::Index> ptr) {
  // TODO: avoid usage of RTTI and dynamic cast
  if (dynamic_cast<T*>(ptr.get()) == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<T>(static_cast<T*>(ptr.release()));
}

inline void check_index_pq_compatibility(const faiss::IndexPQ* pq) {
  U_THROW_IF_MSG(pq->do_polysemous_training,
                 "UNITY error: IndexPQ must not have polysemous training enabled");
  U_THROW_IF_MSG(pq->metric_type != faiss::MetricType::METRIC_L2,
                 "UNITY error: only METRIC_L2 is supported");
}

inline std::unique_ptr<faiss::IndexPQ> read_index_pq(const char* fname) {
  std::unique_ptr<faiss::Index> index(faiss::unity_read_index(fname));
  std::unique_ptr<faiss::IndexPQ> index_pq = dyn_cast<faiss::IndexPQ>(std::move(index));
  if (index_pq == nullptr) {
    U_THROW_MSG(
        "failed to cast index to faiss::IndexPQ*: the provided index file may not be a valid "
        "file of IndexPQ.");
  }
  check_index_pq_compatibility(index_pq.get());
  return index_pq;
}

}  // namespace detail
}  // namespace unity