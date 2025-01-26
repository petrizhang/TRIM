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

#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#include <string>

#include "unity/common/common.h"
#include "unity/common/constants.h"
#include "unity/detail/io/read_hnswlib.h"
#include "unity/detail/io/read_pq.h"
#include "unity/detail/searcher/hnsw/hnsw_searcher.h"
#include "unity/util/thread_pool.h"

namespace unity {
namespace detail {

std::unique_ptr<UnityHnsw> read_uhnsw(const Dict& options) {
  namespace constants = unity::constants;
  std::string metric = options.require<std::string>(constants::U_METRIC);
  U_THROW_IF_NOT_MSG(metric == constants::U_METRIC_L2, "only L2 metric is supported now");
  std::unique_ptr<UnityHnsw> uhnsw = std::make_unique<UnityHnsw>();

  // Read HNSW index
  int dim = options.require<int>(constants::U_DIM);
  std::string hnswlib_index_path = options.require<std::string>(constants::U_HNSWLIB_INDEX_PATH);

  uhnsw->owned_space = std::make_unique<hnswlib::L2Space>(dim);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw = read_hnswlib(
      uhnsw->owned_space.get(), constants::metric_map(metric), hnswlib_index_path, dim);
  uhnsw->owned_index_hnsw = std::move(index_hnsw);

  // Read PQ index
  std::optional<std::string> opt_pq_index_path =
      options.optional<std::string>(constants::U_PQ_INDEX_PATH);
  if (opt_pq_index_path.has_value()) {
    std::string pq_index_path = opt_pq_index_path.value();
    std::optional<int> opt_num_threads = options.optional<int>(constants::U_NUM_THREADS);
    ctpl::thread_pool pool(opt_num_threads.value_or(16));

    std::unique_ptr<faiss::IndexPQ> faiss_index_pq = read_index_pq(pq_index_path.c_str());
    U_THROW_IF_NOT_MSG(faiss_index_pq->pq.nbits == 8, "only support 8bit PQ");

    uhnsw->unity_index_pq = UnityIndexPQ(std::move(faiss_index_pq));
    // Rorder PQ codes
    uhnsw->reorder_pq_codes();
    // Compute PQ reconstruction errors
    ExactDco<> dco(uhnsw->owned_index_hnsw.get());
    uhnsw->unity_index_pq.compute_pq_reconstruction_errors(&dco, pool);
  }

  return uhnsw;
}

std::unique_ptr<ISearcher> create_hnsw_searcher(const Dict& options) {
  std::shared_ptr<UnityHnsw> index = read_uhnsw(options);
  std::unique_ptr<ISearcher> searcher = nullptr;
  bool enable_profile = options.optional<bool>(constants::U_ENABLE_PROFILE).value_or(false);
  std::string dco_type =
      options.optional<std::string>(constants::U_DCO).value_or(constants::U_DCO_UNITY);
  if (dco_type == constants::U_DCO_EXACT) {
    if (enable_profile) {
      ExactDco<true> dco(index.get());
      return std::make_unique<HNSWSearcher<decltype(dco)>>(index, std::move(dco));
    } else {
      ExactDco<false> dco(index.get());
      return std::make_unique<HNSWSearcher<decltype(dco)>>(index, std::move(dco));
    }
  } else if (dco_type == constants::U_DCO_UNITY) {
    U_THROW_IF_NOT_MSG(index->unity_index_pq.owned_index_pq != nullptr,
                       "using UNITY for distance comparision but missing PQ index");
    if (enable_profile) {
      UnityOp8<true> dco(index.get());
      return std::make_unique<HNSWSearcher<decltype(dco)>>(index, std::move(dco));
    } else {
      UnityOp8<false> dco(index.get());
      return std::make_unique<HNSWSearcher<decltype(dco)>>(index, std::move(dco));
    }
  } else {
    U_THROW_FMT("unknown DCO %s", dco_type.c_str());
  }

  return searcher;
}

}  // namespace detail

struct SearcherCreator {
  Dict options;
  std::string index_type;

  explicit SearcherCreator(const std::string& index_type) : index_type(index_type) {}

  SearcherCreator& set(const std::string& key, const Object& value) {
    options.put(key, value);
    return *this;
  }

  std::unique_ptr<ISearcher> create() {
    using unity::constants::U_HNSW;
    if (index_type == U_HNSW) {
      return detail::create_hnsw_searcher(options);
    }
    U_THROW_FMT("cannot create searcher for unsupported index type %s", index_type.c_str());
  }
};

}  // namespace unity