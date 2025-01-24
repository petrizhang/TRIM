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

#include "unity/detail/faiss/MetricType.h"
#include "unity/detail/faiss/impl/io.h"
#include "unity/detail/faiss/impl/io_macros.h"
#include "unity/detail/quantizer/index_pq.h"

namespace unity {
namespace detail {

using namespace faiss;

inline void read_index_header(Index* idx, IOReader* f) {
  READ1(idx->d);
  READ1(idx->ntotal);
  idx_t dummy;
  READ1(dummy);
  READ1(dummy);
  READ1(idx->is_trained);
  READ1(idx->metric_type);
  if (idx->metric_type > 1) {
    READ1(idx->metric_arg);
  }
  idx->verbose = false;
}

inline void read_product_quantizer(ProductQuantizer* pq, IOReader* f) {
  READ1(pq->d);
  READ1(pq->M);
  READ1(pq->nbits);
  pq->set_derived_values();
  READVECTOR(pq->centroids);
}

inline std::unique_ptr<Index> read_index(IOReader* f) {
  uint32_t h;
  READ1(h);
  if (h == fourcc("IxPQ") || h == fourcc("IxPo") || h == fourcc("IxPq")) {
    // IxPQ and IxPo were merged into the same IndexPQ object
    std::unique_ptr<IndexPQ> idxp = std::make_unique<IndexPQ>();
    read_index_header(idxp.get(), f);
    read_product_quantizer(&idxp->quantizer, f);
    idxp->code_size = idxp->quantizer.code_size;
    READVECTOR(idxp->codes);
    if (h == fourcc("IxPo") || h == fourcc("IxPq")) {
      READ1(idxp->search_type);
      READ1(idxp->encode_signs);
      READ1(idxp->polysemous_ht);
    }
    // Old versions of PQ all had metric_type set to INNER_PRODUCT
    // when they were in fact using L2. Therefore, we force metric type
    // to L2 when the old format is detected
    if (h == fourcc("IxPQ") || h == fourcc("IxPo")) {
      idxp->metric_type = METRIC_L2;
    }

    idxp->index_type = IndexType::INDEX_PQ;
    return idxp;
  }

  FAISS_ASSERT_FMT(false, "unknown file format: %s", f->name.c_str());
}

inline std::unique_ptr<Index> read_index(FILE* f) {
  FileIOReader reader(f);
  return read_index(&reader);
}

inline std::unique_ptr<Index> read_index(const char* fname) {
  FileIOReader reader(fname);
  return read_index(&reader);
}

template <IndexType t>
struct IndexTypeDispatch;

template <>
struct IndexTypeDispatch<INDEX_PQ> {
  using type = IndexPQ;
};

template <IndexType t>
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

}  // namespace detail
}  // namespace unity
