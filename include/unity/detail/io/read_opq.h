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

#include "faiss/IndexPreTransform.h"
#include "faiss/VectorTransform.h"
#include "unity/common/u_assert.h"
#include "unity/detail/index/opq.h"
#include "unity/detail/io/read_faiss.h"
#include "unity/detail/io/read_pq.h"

namespace unity {
namespace detail {

inline void check_opq_compatibility(const faiss::IndexPreTransform* transform,
                                    const faiss::IndexPQ* pq) {
  U_THROW_IF_NOT_MSG(transform->chain.size() == 1,
                     "reading OPQ error: OPQ index only allows one transform");
  U_THROW_IF_NOT_MSG(dynamic_cast<faiss::LinearTransform*>(transform->chain[0]) != nullptr,
                     "reading OPQ error: only faiss::OPQMatrix transform is allowed");
  U_THROW_IF_NOT_MSG(!pq->do_polysemous_training,
                     "UNITY error: IndexPQ must not have polysemous training enabled");
  U_THROW_IF_NOT_MSG(pq->metric_type == faiss::MetricType::METRIC_L2,
                     "UNITY error: only METRIC_L2 is supported");
}

inline UnityIndexOPQ read_index_opq(const char* fname) {
  std::unique_ptr<faiss::Index> index(faiss::unity_read_index(fname));
  std::unique_ptr<faiss::IndexPreTransform> index_pre_transform =
      dyn_cast<faiss::IndexPreTransform>(std::move(index));
  U_THROW_IF_NOT_MSG(index_pre_transform != nullptr,
                     "error reading OPQ: failed to cast faiss::Index* to "
                     "faiss::IndexPreTransform*");

  faiss::IndexPQ* index_pq = dynamic_cast<faiss::IndexPQ*>(index_pre_transform->index);
  U_THROW_IF_NOT_MSG(
      index_pq != nullptr,
      "error reading OPQ: failed to extract faiss::IndexPQ from faiss::IndexPreTransform");

  check_opq_compatibility(index_pre_transform.get(), index_pq);

  UnityIndexOPQ index_opq(std::move(index_pre_transform), index_pq,
                          index_pre_transform.get());
  return std::move(index_opq);
}

}  // namespace detail
}  // namespace unity