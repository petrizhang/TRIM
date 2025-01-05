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

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#include "top/common/deo.h"
#include "top/detail/faiss/impl/ProductQuantizer.h"
#include "top/detail/faiss/impl/code_distance/code_distance.h"
#include "top/detail/hnswlib/hnswlib.h"
#include "top/detail/quantization/index_pq.h"

namespace top {
namespace detail {

template <typename PQEncoderType>
struct TopDEO : DistanceEstimateOperator<float, TopDEO<PQEncoderType>> {
  using dist_type = float;

  const IndexPQ* pq = nullptr;
  const hnswlib::HierarchicalNSW<float>* hnsw = nullptr;
  const dist_type* query = nullptr;
  AlignedTable<dist_type> dist_table;

  TopDEO() = default;
  TopDEO(const IndexPQ* pq, const hnswlib::HierarchicalNSW<float>* hnsw) : pq(pq), hnsw(hnsw) {}

  void set_query_impl(const dist_type* query_data) {
    query = query_data;
    dist_table.resize(pq->pq.M * pq->pq.ksub);
    pq->pq.compute_distance_table(query, dist_table.data());
  }

  void prefetch_impl(int i) {
#ifdef __SSE__
    _mm_prefetch((char*)(pq->codes.data() + pq->code_size * i), _MM_HINT_T0);
#endif
  }

  float estimate_impl(int i) {
    return distance_single_code<PQEncoderType>(pq->pq.M, pq->pq.nbits, dist_table.data(),
                                               pq->codes.data() + pq->code_size * i);
  }

  float compute_impl(int i) { return 0; }
};

using TopDEO8 = TopDEO<faiss::PQDecoder8>;
using TopDEO16 = TopDEO<faiss::PQDecoder16>;
using TopDEOGeneric = TopDEO<faiss::PQDecoderGeneric>;

}  // namespace detail
}  // namespace top