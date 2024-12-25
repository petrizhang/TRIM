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

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "top/hnsw/HNSWInitializer.h"
#include "top/core/memory.h"
#include "top/core/distance.h"

namespace top {

constexpr int EMPTY_ID = -1;

template <typename node_t>
struct Graph {
  int N, K;

  node_t* data = nullptr;

  std::unique_ptr<HNSWInitializer> initializer = nullptr;

  std::vector<int> eps;

  Graph() = default;

  Graph(node_t* edges, int N, int K) : N(N), K(K), data(edges) {}

  Graph(int N, int K) : N(N), K(K), data((node_t*)alloc2M((size_t)N * K * sizeof(node_t))) {}

  Graph(const Graph& g) : Graph(g.N, g.K) {
    this->eps = g.eps;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < K; ++j) {
        at(i, j) = g.at(i, j);
      }
    }
    if (g.initializer) {
      initializer = std::make_unique<HNSWInitializer>(*g.initializer);
    }
  }

  void init(int N, int K) {
    data = (node_t*)alloc2M((size_t)N * K * sizeof(node_t));
    std::memset(data, -1, N * K * sizeof(node_t));
    this->K = K;
    this->N = N;
  }

  ~Graph() { free(data); }

  const int* edges(int u) const { return data + K * u; }

  int* edges(int u) { return data + K * u; }

  node_t at(int i, int j) const { return data[i * K + j]; }

  node_t& at(int i, int j) { return data[i * K + j]; }

  void prefetch(int u, int lines) const { mem_prefetch((char*)edges(u), lines); }

  template <typename Pool, typename Computer>
  void initialize_search(Pool& pool, const Computer& computer) const {
    if (initializer) {
      initializer->initialize(pool, computer);
    } else {
      for (auto ep : eps) {
        pool.insert(ep, computer(ep));
      }
    }
  }
};

}  // namespace top