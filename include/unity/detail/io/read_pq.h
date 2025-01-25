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

#include "faiss/Index.h"

namespace unity {

template <IndexType t>
struct IndexTypeDispatch;

template <>
struct IndexTypeDispatch<INDEX_PQ> {
  using type = IndexPQ;
};

template <faiss::IndexType t>
inline std::unique_ptr<typename IndexTypeDispatch<t>::type> checked_cast(
    std::unique_ptr<Index> ptr) {
  using ChildType = typename IndexTypeDispatch<t>::type;
  FAISS_THROW_IF_NOT(ptr->index_type == t);
  return std::unique_ptr<ChildType>(static_cast<ChildType*>(ptr.release()));
}

inline void check_index_pq_compatibility(const IndexPQ* pq) {
  FAISS_THROW_IF_MSG(pq->do_polysemous_training,
                     "UNITY error: IndexPQ must not have polysemous training enabled");
  FAISS_THROW_IF_MSG(pq->metric_type != MetricType::METRIC_L2,
                     "UNITY error: only METRIC_L2 is supported");
}

inline std::unique_ptr<IndexPQ> read_index_pq(const char* fname) {
  std::unique_ptr<Index> index = detail::read_index(fname);
  std::unique_ptr<IndexPQ> index_pq = detail::checked_cast<IndexType::INDEX_PQ>(std::move(index));
  detail::check_index_pq_compatibility(index_pq.get());
  return index_pq;
}

}