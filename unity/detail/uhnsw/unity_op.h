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

#include "unity/common/dco.h"
#include "unity/detail/faiss/impl/ProductQuantizer.h"
#include "unity/detail/faiss/impl/code_distance/code_distance.h"
#include "unity/detail/hnswlib/hnswlib.h"
#include "unity/detail/uhnsw/unity_hnsw.h"

namespace unity {
namespace detail {

template <typename PQEncoderType, bool enable_profile = false>
struct UnityOp final : IDistanceComparisonOperator<unsigned, float> {
  using idx_t = unsigned;
  using dist_t = float;

  AlignedTable<dist_t> dist_table;
  const IndexPQ* pq = nullptr;
  const hnswlib::HierarchicalNSW<float>* hnsw = nullptr;
  const dist_t* query = nullptr;
  float gamma = 1;

  ~UnityOp() override = default;

  explicit UnityOp(const UnityHNSW* u_hnsw) {
    U_ASSERT(u_hnsw != nullptr);
    U_ASSERT(u_hnsw->owned_index_hnsw != nullptr);
    U_ASSERT(u_hnsw->owned_index_pq != nullptr);
    U_ASSERT(u_hnsw->owned_space != nullptr);
    pq = u_hnsw->owned_index_pq.get();
    hnsw = u_hnsw->owned_index_hnsw.get();
  }

  virtual void set_query(const dist_t* query_data) override {
    query = query_data;
    dist_table.resize(pq->pq.M * pq->pq.ksub);
    pq->pq.compute_distance_table(query, dist_table.data());
  }

  void prefetch_pq_codes(unsigned i) {
#ifdef __SSE__
    _mm_prefetch((char*)(pq->codes.data() + pq->code_size * i), _MM_HINT_T0);
#endif
  }

  virtual bool distance_less_than(dist_t max_dist, idx_t i, float* dist) const override {
    return true;
  }

  virtual bool distance4_less_than(dist_t max_dist, idx_t i0, idx_t i1, idx_t i2, idx_t i3,
                                   ComparisonResult4<dist_t>* __restrict result) const override {
    return true;
  }
};

template <bool enable_profile>
using UnityOp8 = UnityOp<faiss::PQDecoder8, enable_profile>;

}  // namespace detail
}  // namespace unity