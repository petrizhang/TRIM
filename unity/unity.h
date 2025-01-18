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

#include <string>

#include "unity/common/common.h"
#include "unity/common/constants.h"
#include "unity/detail/hnsw/hnsw_searcher.h"
#include "unity/detail/io/read_faiss.h"
#include "unity/detail/io/read_hnswlib.h"
#include "unity/util/thread_pool.h"

namespace unity {
namespace detail {

std::unique_ptr<Searcher> build_hnsw_searcher(const Dict& options) {
  namespace constants = unity::constants;

  int dim = options.require<int>(constants::U_DIM);
  std::string hnswlib_index_path = options.require<std::string>(constants::U_HNSWLIB_INDEX_PATH);
  std::string pq_index_path = options.require<std::string>(constants::U_PQ_INDEX_PATH);
  std::string metric = options.require<std::string>(constants::U_METRIC);
  int num_threads = options.require<int>(constants::U_NUM_THREADS);

  TOP_THROW_IF_NOT_MSG(metric == constants::U_METRIC_L2, "only L2 metric is supported now");
  auto searcher = std::make_unique<HNSWSearcher>();

  // Read HNSW index
  ctpl::thread_pool pool(num_threads);
  searcher->owned_space = std::make_unique<hnswlib::L2Space>(dim);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw = read_hnswlib(
      searcher->owned_space.get(), constants::metric_map(metric), hnswlib_index_path, dim);
  searcher->owned_index_hnsw = std::move(index_hnsw);

  // Read PQ index
  std::unique_ptr<IndexPQ> index_pq = read_index_pq(pq_index_path.c_str());
  TOP_THROW_IF_NOT_MSG(index_pq->pq.nbits == 8, "only support 8bit PQ");
  searcher->owned_index_pq = std::move(index_pq);
  // Rorder PQ codes
  searcher->_reorder_pq_codes();
  // Compute PQ reconstruction errors
  searcher->_compute_pq_reconstruction_errors(pool);
  // Set DEO
  searcher->owned_deo =
      std::make_unique<TopDEO8>(searcher->owned_index_pq.get(), searcher->owned_index_hnsw.get());
  return searcher;
}

}  // namespace detail

struct SearcherCreator {
  Dict options;
  std::string index_type;

  SearcherCreator(const std::string& index_type) : index_type(index_type) {}

  SearcherCreator& set(const std::string& key, const Object& value) {
    options.put(key, value);
    return *this;
  }

  std::unique_ptr<Searcher> build() {
    using unity::constants::U_HNSW;
    if (index_type == U_HNSW) {
      return detail::build_hnsw_searcher(options);
    }
    TOP_THROW_FMT("cannot create searcher for unsupported index type %s", index_type.c_str());
  }
};

}  // namespace unity