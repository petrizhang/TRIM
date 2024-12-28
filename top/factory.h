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
#include "top/detail/io/read_hnswlib.h"
#include "top/detail/searcher/hnsw_searcher.h"

namespace top {
namespace detail {

/**
 * Requried options: `dim`, `metric`
 * Optional options: `ef`, `use_bounded_queue`,
 */
std::unique_ptr<Searcher> build_hnsw_searcher(const Dict& options) {
  using std::string;
  using top::constants::TOP_DIM;
  using top::constants::TOP_EF;
  using top::constants::TOP_HNSWLIB_INDEX_PATH;
  using top::constants::TOP_METRIC;
  using top::constants::TOP_METRIC_L2;

  int dim = options.require<int>(TOP_DIM);
  string hnswlib_index_path = options.require<string>(TOP_HNSWLIB_INDEX_PATH);
  string metric = options.require<string>(TOP_METRIC);

  if (metric != TOP_METRIC_L2) {
    TOP_THROW_MSG("only L2 metric is supported now");
  }

  auto m = top::constants::metric_map(metric);
  Graph<int> graph = read_hnswlib(m, hnswlib_index_path, dim);
  return std::make_unique<HNSWSearcher>(graph, FP32Quantizer());
}

}  // namespace detail

struct SearcherBuilder {
  Dict options;
  std::string index_type;

  SearcherBuilder(const std::string& index_type) : index_type(index_type) {}

  SearcherBuilder& set(const std::string& name, const Object& value) {
    options.put(name, value);
    return *this;
  }

  std::unique_ptr<Searcher> build() {
    using top::constants::TOP_HNSW;
    if (index_type == TOP_HNSW) {
      return detail::build_hnsw_searcher(options);
    }
    TOP_THROW_FMT("cannot create searcher for unsupported index type %s", index_type);
  }
};

}  // namespace top