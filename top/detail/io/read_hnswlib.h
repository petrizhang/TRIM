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

#include "top/detail/hnsw/graph.h"
#include "top/detail/hnswlib/hnswalg.h"
#include "top/detail/hnswlib/space_l2.h"

namespace top {
namespace detail {

Graph<int> read_hnswlib(const std::string& path, int dim) {
  auto space = std::make_unique<hnswlib::L2Space>(dim);
  auto hnsw = std::make_unique<hnswlib::HierarchicalNSW<float>>(space.get(), path, false, 0, false);
  int64_t nb = hnsw->cur_element_count;

  Graph<int> final_graph;
  int M = hnsw->M_;
  final_graph.init(nb, hnsw->M_);

  // TODO: parallelize it
  for (int i = 0; i < nb; ++i) {
    int* edges = (int*)hnsw->get_linklist0(i);
    for (int j = 1; j <= edges[0]; ++j) {
      final_graph.at(i, j - 1) = edges[j];
    }
  }

  auto initializer = std::make_unique<HNSWInitializer>(nb, M);
  initializer->ep = hnsw->enterpoint_node_;
  for (int i = 0; i < nb; ++i) {
    int level = hnsw->element_levels_[i];
    initializer->levels[i] = level;
    if (level > 0) {
      initializer->lists[i].assign(level * M, -1);
      for (int j = 1; j <= level; ++j) {
        int* edges = (int*)hnsw->get_linklist(i, j);
        for (int k = 1; k <= edges[0]; ++k) {
          initializer->at(j, i, k - 1) = edges[k];
        }
      }
    }
  }

  final_graph.initializer = std::move(initializer);
  return final_graph;
}

}  // namespace detail
}  // namespace top
