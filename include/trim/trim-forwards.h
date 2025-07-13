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

#include "trim/common/common.h"
#include "trim/common/constants.h"
#include "trim/detail/io/read_hnswlib.h"
#include "trim/detail/io/read_opq.h"
#include "trim/detail/io/read_pq.h"
#include "trim/detail/searcher/hnsw/hnsw_searcher.h"
#include "trim/detail/searcher/ivfpq_searcher.h"
#include "trim/util/thread_pool.h"

namespace trim {
namespace detail {

std::unique_ptr<TrimHNSW> read_uhnsw(const Dict& options) {
  namespace constants = trim::constants;
  std::string metric = options.require<std::string>(constants::T_METRIC);
  T_THROW_IF_NOT_MSG(metric == constants::T_METRIC_L2, "only L2 metric is supported now");
  std::unique_ptr<TrimHNSW> uhnsw = std::make_unique<TrimHNSW>();

  // Read HNSW index
  int dim = options.require<int>(constants::T_DIM);
  std::string hnswlib_index_path = options.require<std::string>(constants::T_HNSWLIB_INDEX_PATH);

  uhnsw->owned_space = std::make_unique<hnswlib::L2Space>(dim);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw = read_hnswlib(
      uhnsw->owned_space.get(), constants::metric_map(metric), hnswlib_index_path, dim);
  uhnsw->owned_index_hnsw = std::move(index_hnsw);

  // Read PQ index
  std::optional<std::string> opt_pq_index_path =
      options.optional<std::string>(constants::T_PQ_INDEX_PATH);
  if (opt_pq_index_path.has_value()) {
    std::string pq_index_path = opt_pq_index_path.value();

    std::optional<bool> opt_use_opq = options.optional<bool>(constants::T_USE_OPQ);
    bool use_opq = opt_use_opq.value_or(false);
    std::optional<int> opt_num_threads = options.optional<int>(constants::T_NUM_THREADS);
    ctpl::thread_pool pool(opt_num_threads.value_or(16));

    if (!use_opq) {
      std::unique_ptr<faiss::IndexPQ> faiss_index_pq = read_index_pq(pq_index_path.c_str());
      T_THROW_IF_NOT_MSG(faiss_index_pq->pq.nbits == 8, "only support 8bit PQ");
      uhnsw->trim_index_opq =
          TrimIndexOPQ(std::move(faiss_index_pq), faiss_index_pq.get(), nullptr);
    } else {
      TrimIndexOPQ index_opq = read_index_opq(pq_index_path.c_str());
      uhnsw->trim_index_opq = std::move(index_opq);
    }

    // Rorder PQ codes
    uhnsw->reorder_pq_codes();
    // Compute PQ reconstruction errors
    ExactDCO<> dco(uhnsw->owned_index_hnsw.get());
    uhnsw->trim_index_opq.compute_pq_reconstruction_errors(&dco, pool);
  }

  return uhnsw;
}

std::unique_ptr<ISearcher> create_hnsw_searcher(const Dict& options) {
  std::shared_ptr<TrimHNSW> index = read_uhnsw(options);
  std::unique_ptr<ISearcher> searcher = nullptr;

  bool enable_profile = options.optional<bool>(constants::T_ENABLE_PROFILE).value_or(false);
  std::string dco_type = options.optional<std::string>(constants::T_DCO).value_or(constants::T_DCO_TRIM);
  std::size_t rl_size = options.optional<size_t>(constants::RANDOM_LANDMARK_SIZE).value_or(0);
  std::cout<< "random_landmark_size: " << rl_size << std::endl;

  if (dco_type == constants::T_DCO_EXACT) {
    if (enable_profile) {
      ExactDCO<true> dco(index.get());
      return std::make_unique<HNSWSearcher<decltype(dco)>>(index, std::move(dco));
    } else {
      ExactDCO<false> dco(index.get());
      return std::make_unique<HNSWSearcher<decltype(dco)>>(index, std::move(dco));
    }
  } else if (dco_type == constants::T_DCO_TRIM) {
    T_THROW_IF_NOT_MSG(index->trim_index_opq.pq.index_pq != nullptr,
                       "using TRIM for distance comparision but missing PQ index");
    if (enable_profile) {
      TrimDCO8<true> dco(index.get(), rl_size);
      return std::make_unique<HNSWSearcher<decltype(dco)>>(index, std::move(dco));
    } else {
      TrimDCO8<false> dco(index.get(), rl_size);
      return std::make_unique<HNSWSearcher<decltype(dco)>>(index, std::move(dco));
    }
  } else {
    T_THROW_FMT("unknown DCO %s", dco_type.c_str());
  }

  return searcher;
}

std::unique_ptr<ISearcher> create_ivfpq_searcher(const Dict& options) {
  std::string index_path = options.require<std::string>(constants::T_IVFPQ_INDEX_PATH);
  // std::string data_path = options.require<std::string>(constants::DATA_PATH);
  std::unique_ptr<ISearcher> searcher = std::make_unique<IVFPQSearcher>(index_path.c_str());
    
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
    using trim::constants::T_HNSW;
    using trim::constants::T_IVFPQ;
    if (index_type == T_HNSW) {
      return detail::create_hnsw_searcher(options);
    } 
    else if (index_type == T_IVFPQ) {
      return detail::create_ivfpq_searcher(options);
    }
    T_THROW_FMT("cannot create searcher for unsupported index type %s", index_type.c_str());
  }
};

}  // namespace trim