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

#include "top/common/common.h"
#include "top/common/constants.h"
#include "top/detail/hnsw/hnsw_searcher.h"
#include "top/detail/io/read_hnswlib.h"

namespace top {
namespace detail {

/**
 * Requried options: `dim`, `metric`
 * Optional options: `ef`, `use_bounded_queue`,
 */
std::unique_ptr<Searcher> build_hnsw_searcher(const Dict& options) {
  namespace constants = top::constants;

  int dim = options.require<int>(constants::TOP_DIM);
  std::string hnswlib_index_path = options.require<std::string>(constants::TOP_HNSWLIB_INDEX_PATH);
  std::string metric = options.require<std::string>(constants::TOP_METRIC);
  if (metric != constants::TOP_METRIC_L2) {
    TOP_THROW_MSG("only L2 metric is supported now");
  }
  auto searcher = std::make_unique<HNSWSearcher>();

  searcher->owned_space = std::make_unique<hnswlib::L2Space>(dim);
  auto m = constants::metric_map(metric);
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index =
      read_hnswlib(searcher->owned_space.get(), m, hnswlib_index_path, dim);
  searcher->owned_index = std::move(index);
  return searcher;
}

}  // namespace detail

struct SearcherBuilder {
  Dict options;
  std::string index_type;

  SearcherBuilder(const std::string& index_type) : index_type(index_type) {}

  SearcherBuilder& set(const std::string& key, const Object& value) {
    options.put(key, value);
    return *this;
  }

  std::unique_ptr<Searcher> build() {
    using top::constants::TOP_HNSW;
    if (index_type == TOP_HNSW) {
      return detail::build_hnsw_searcher(options);
    }
    TOP_THROW_FMT("cannot create searcher for unsupported index type %s", index_type.c_str());
  }
};

}  // namespace top