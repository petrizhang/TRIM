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

#include "top/f32_quant.h"
#include "top/hnsw/graph.h"
#include "top/neighbor.h"
#include "top/searcher.h"

namespace top {

struct HNSWSearcher : Searcher {
  int d;                   ///< vector dimension
  int nb;                  ///< number of base vectors
  const Graph<int> graph;  ///< the HNSW graph
  FP32Quantizer quant;
  // Search parameters
  int ef = 32;  ///< search parameter
  bool use_bounded_queue = false;
  // Profile
  bool enable_profile = false;
  // Memory prefetch parameters
  int po = 1;
  int pl = 1;
  const int graph_po;

  explicit HNSWSearcher(const Graph<int>& graph) : graph(graph), graph_po(graph.K / 16) {}
  ~HNSWSearcher() override = default;

  void set_data(const float* data, int n, int dim) override;
  void ann_search(const float* q, int k, int* dst) const override;
  void range_search(const float* q, float radius, int* dst) const override;
  void set(const std::string& key, Object value) override;
  void optimize(int num_threads) override;
  Dict get_profile() const override;

  template <typename Computer>
  void ann_search_bounded(searcher::LinearPool<float>& pool, const Computer& computer) const;

  template <typename Computer>
  void ann_search_unbounded(searcher::LinearPool<float>& pool, const Computer& computer) const;
};

inline void HNSWSearcher::set_data(const float* data, int n, int dim) {}

inline void HNSWSearcher::ann_search(const float* q, int k, int* dst) const {
  const auto computer = quant.get_computer(q);
  searcher::LinearPool<float> pool(nb, std::max(k, ef), k);
  graph.initialize_search(pool, computer);
  ann_search_bounded(pool, computer);
  quant.reorder(pool, q, dst, k);
}

inline void HNSWSearcher::range_search(const float* q, float radius, int* dst) const {
  FAISS_THROW_MSG("not implemented error");
}

inline void HNSWSearcher::set(const std::string& key, Object value) {
  if (key == "ef") {
    FAISS_THROW_IF_NOT_MSG(value.type == ObjectType::INTEGER_TYPE, "ef must be an integer");
    ef = static_cast<int>(value.get_integer());
  }

  FAISS_THROW_FMT("unknown parameter %s", key.c_str());
}

inline void HNSWSearcher::optimize(int num_threads) { FAISS_THROW_MSG("not implemented error"); }

inline Dict HNSWSearcher::get_profile() const { return {}; }

template <typename Computer>
void HNSWSearcher::ann_search_bounded(searcher::LinearPool<float>& pool,
                                      const Computer& computer) const {
  while (pool.has_next()) {
    auto u = pool.pop();
    graph.prefetch(u, graph_po);
    for (int i = 0; i < po; ++i) {
      int to = graph.at(u, i);
      computer.prefetch(to, pl);
    }
    for (int i = 0; i < graph.K; ++i) {
      int v = graph.at(u, i);
      if (v == -1) {
        break;
      }
      if (i + po < graph.K && graph.at(u, i + po) != -1) {
        int to = graph.at(u, i + po);
        computer.prefetch(to, pl);
      }
      if (pool.vis.get(v)) {
        continue;
      }
      pool.vis.set(v);
      auto cur_dist = computer(v);
      pool.insert(v, cur_dist);
    }
  }
}

template <typename Computer>
void HNSWSearcher::ann_search_unbounded(searcher::LinearPool<float>& seeds,
                                        const Computer& computer) const {
  searcher::UnboundedMaxHeap<float> results;
  searcher::UnboundedMaxHeap<float> candidates;
     searcher::Bitset<> visited(nb);

  for (auto& neighbor : seeds.data_) {
    results.emplace(neighbor.id, neighbor.distance);
    candidates.emplace(neighbor.id, -neighbor.distance);
  }

  while (!candidates.empty()) {
    const searcher::Neighbor<> nearest_cand = candidates.top();
    const float furthest_d = results.top().distance;
    int u = nearest_cand.id;
    const float cand_d = -nearest_cand.distance;
    if (furthest_d < cand_d) {
      break;
    }
    candidates.pop();

    graph.prefetch(u, graph_po);
    for (int i = 0; i < po; ++i) {
      int to = graph.at(u, i);
      computer.prefetch(to, pl);
    }

    for (int i = 0; i < graph.K; ++i) {
      int v = graph.at(u, i);
      if (v == -1) {
        break;
      }

      if (i + po < graph.K && graph.at(u, i + po) != -1) {
        int to = graph.at(u, i + po);
        computer.prefetch(to, pl);
      }

      if (visited.get(v)) {
        continue;
      }
      visited.set(v);
      auto cur_dist = computer(v);

      if (cur_dist < cand_d || results.size() < ef) {
        candidates.emplace(v, -cur_dist);
        results.emplace(v, -cur_dist);
        if (results.size() > ef) {
          results.pop();
        }
      }
    }
  }
}
}  // namespace top