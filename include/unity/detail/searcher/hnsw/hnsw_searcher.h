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

#include <algorithm>
#include <memory>
#include <queue>
#include <type_traits>

#include "unity/common/common.h"
#include "unity/common/prefetch.h"
#include "unity/common/searcher.h"
#include "unity/common/setter_proxy.h"
#include "unity/detail/index/hnsw.h"
#include "unity/detail/searcher/hnsw/dco_exact.h"
#include "unity/detail/searcher/hnsw/dco_unity.h"

namespace unity {
namespace detail {

template <typename TDco = UnityDCO8<false>>
struct HNSWSearcher : SetterProxy<HNSWSearcher<TDco>>, ISearcher {
  static_assert(std::is_base_of_v<IDCO, TDco>, "Error: DCO must inherit from IDCO.");
  using dist_t = float;
  using idx_t = unsigned;
  using This = HNSWSearcher<TDco>;
  using Proxy = SetterProxy<This>;
  using CompareByFirst = HnswlibIndex::CompareByFirst;
  using VisitedList = hnswlib::VisitedList;
  using labeltype = hnswlib::labeltype;
  using tableint = hnswlib::tableint;
  using vl_type = hnswlib::vl_type;
  using linklistsizeint = hnswlib::linklistsizeint;
  using ResultQueue =
      std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                          HnswlibIndex::CompareByFirst>;

  // Distance comparison operator used during search
  mutable TDco _dco;

  // HNSW members
  hnswlib::VisitedListPool* _visited_list_pool{nullptr};
  char* _data_level0_memory{nullptr};
  size_t _size_data_per_element{0};
  size_t _size_links_per_element{0};
  size_t _offset_data{0}, _offset_level0{0};

  // Index
  const HnswlibIndex* _hnsw{nullptr};
  std::shared_ptr<UnityHnsw> _shared_uhnsw{nullptr};  // a UnityHnsw index with shared ownership
  const UnityHnsw* _uhnsw{nullptr};                   // the underlying pointer of [shared_uhnsw]

  // Parameters
  size_t _ef{32};
  size_t _refine_queue_size{0};
  bool _enable_profile{false};
  bool _enable_batch_dco{true};

  U_FORBID_COPY_AND_ASSIGN(HNSWSearcher);

  U_FORBID_MOVE(HNSWSearcher);

  HNSWSearcher() = delete;

  explicit HNSWSearcher(const std::shared_ptr<UnityHnsw>& index, TDco dco)
      : Proxy("HNSWSearcher"), _dco(std::move(dco)), _shared_uhnsw(index), _uhnsw(index.get()) {
    U_ASSERT(_uhnsw != nullptr);
    _hnsw = _uhnsw->owned_index_hnsw.get();
    U_ASSERT(_hnsw != nullptr);
    _visited_list_pool = _hnsw->visited_list_pool_.get();
    _data_level0_memory = _hnsw->data_level0_memory_;
    _size_data_per_element = _hnsw->size_data_per_element_;
    _size_links_per_element = _hnsw->size_links_per_element_;
    _offset_data = _hnsw->offsetData_;
    _offset_level0 = _hnsw->offsetLevel0_;

    Proxy::template bind<INTEGER_TYPE>("ef", &This::set_ef);
    Proxy::template bind<INTEGER_TYPE>("refine_queue_size", &This::set_refine_queue_size);
    Proxy::template bind<BOOL_TYPE>("enable_profile", &This::set_enable_profile);
    Proxy::template bind<BOOL_TYPE>("enable_batch_dco", &This::set_enable_batch_dco);
    Proxy::template bind<DOUBLE_TYPE>("gamma", &This::set_gamma);
  }

  ~HNSWSearcher() override = default;

  void set_data(const float* data, int n, int dim) override {};

  void set(const std::string& key, const Object& value) override { Proxy::proxy_set(key, value); }

  void try_set(const std::string& key, const Object& value) override {
    Proxy::proxy_try_set(key, value);
  }

  const float* get_data(unsigned i) const override {
    if (_uhnsw == nullptr) {
      return nullptr;
    }

    auto iter = _uhnsw->owned_index_hnsw->label_lookup_.find(i);
    if (iter == _uhnsw->owned_index_hnsw->label_lookup_.end()) {
      return nullptr;
    }

    return reinterpret_cast<const float*>(
        _uhnsw->owned_index_hnsw->getDataByInternalId(iter->second));
  }

  size_t num_data_points() const override {
    return _uhnsw == nullptr ? 0 : _uhnsw->owned_index_hnsw->getCurrentElementCount();
  }

  void ann_search(const float* q, int k, int* dst) const override { _ann_search(q, k, dst); }

  void range_search(const float* q, float radius, int* dst) const override {
    U_THROW_MSG("not implemented error");
  }

  void optimize(int num_threads) override { U_THROW_MSG("not implemented error"); }

  IDCO* get_dco() const override { return &_dco; }

  Dict get_profile() const override { return _dco.get_profile(); }

  void set_ef(size_t ef) { _ef = ef; }

  void set_refine_queue_size(size_t refine_queue_size) { _refine_queue_size = refine_queue_size; }

  void set_enable_profile(bool enable_profile) { _enable_profile = enable_profile; }

  void set_enable_batch_dco(bool enable_batch_dco) { _enable_batch_dco = enable_batch_dco; }

  void set_gamma(float gamma) { _dco.try_set("gamma", gamma); }

  tableint _init_search_seed(const void* query_data, dist_t* dist) const {
    const HnswlibIndex& index = *_hnsw;

    tableint node = index.enterpoint_node_;
    dist_t curdist = _dco.compute(node);

    for (int level = index.maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        unsigned int* data;

        data = (unsigned int*)index.get_linklist(node, level);
        int size = index.getListCount(data);

        tableint* datal = (tableint*)(data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand >= index.max_elements_) {
            U_THROW_FMT("search error, got illegal candidcate %u", cand);
          }
          dist_t d;
          if (_dco.dist_comp(curdist, cand, d)) {
            curdist = d;
            node = cand;
            changed = true;
          }
        }
      }
    }
    *dist = curdist;
    return node;
  }

  /// ANN search implementation
  void _ann_search(const void* query_data, size_t k, int* dst) const {
    U_ASSERT(k > 0);
    const HnswlibIndex& index = *_hnsw;
    if (index.cur_element_count.load() == 0) return;

    _dco.set_query((dist_t*)query_data);
    dist_t seed_dist = std::numeric_limits<dist_t>::max();
    tableint seed = _init_search_seed(query_data, &seed_dist);

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        HnswlibIndex::CompareByFirst>
        top_candidates;

    if (_enable_batch_dco) {
      if (_refine_queue_size > 0) {
        // No less than k and no greater than ef
        size_t actual_refine_queue_size = std::min(std::max(k, _refine_queue_size), _ef);
        top_candidates = _ann_search_level0_batch8_with_refine_queue(
            seed, seed_dist, actual_refine_queue_size, std::max(_ef, k));
      } else {
        top_candidates = _ann_search_level0_batch8(seed, seed_dist, std::max(_ef, k));
      }
    } else {
      top_candidates = _ann_search_level0(seed, seed_dist, std::max(_ef, k));
    }

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }

    int n = top_candidates.size();
    while (n > 0) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      dst[n - 1] = index.getExternalLabel(rez.second);
      top_candidates.pop();
      n--;
    }
    return;
  }

  /// Peform ANN search over the bottom layer
  ResultQueue _ann_search_level0(tableint seed, dist_t seed_dist, size_t ef) const {
    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        result_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        search_queue;

    // Init search
    dist_t max_dist = seed_dist;
    result_queue.emplace(max_dist, seed);
    search_queue.emplace(-max_dist, seed);
    visited[seed] = visited_array_tag;

    // Unbounded beam search
    while (!search_queue.empty()) {
      std::pair<dist_t, tableint> current_node_pair = search_queue.top();
      dist_t current_dist = -current_node_pair.first;
      if (current_dist > max_dist) {
        break;
      }
      search_queue.pop();

      tableint current_node_id = current_node_pair.second;
      int* neighbors = (int*)get_linklist0(current_node_id);
      size_t size = get_list_count((linklistsizeint*)neighbors);

      // Prefetch visited array
      prefetch_l1((char*)(visited + *(neighbors + 1)));
      prefetch_l1((char*)(visited + *(neighbors + 1) + 64));
      // Prefetch data of the first neighbor
      _dco.prefetch(*(neighbors + 1));
      // Prefetch the second neighbor
      prefetch_l1((char*)(neighbors + 2));

      for (size_t j = 1; j <= size; j++) {
        int neighbor_id = *(neighbors + j);
        // Prefetch visited array of the next neighbor
        prefetch_l1((char*)(visited + *(neighbors + j + 1)));
        // Prefetch data of the next neighbor
        _dco.prefetch(*(neighbors + j + 1));

        if (!(visited[neighbor_id] == visited_array_tag)) {
          visited[neighbor_id] = visited_array_tag;
          if (UNLIKELY(result_queue.size() < ef)) {
            dist_t dist = _dco.compute(neighbor_id);
            search_queue.emplace(-dist, neighbor_id);
            result_queue.emplace(dist, neighbor_id);
            max_dist = result_queue.top().first;
          } else {
            dist_t dist;
            if (_dco.dist_comp(max_dist, neighbor_id, dist) || result_queue.size() < ef) {
              search_queue.emplace(-dist, neighbor_id);

              // Prefetch neighbor list of the top object in candidate queue
              prefetch_l1(_data_level0_memory + search_queue.top().second * _size_data_per_element +
                          _offset_level0);

              result_queue.emplace(dist, neighbor_id);

              while (result_queue.size() > ef) {
                result_queue.pop();
              }

              if (!result_queue.empty()) max_dist = result_queue.top().first;
            }
          }
        }
      }
    }

    _visited_list_pool->releaseVisitedList(vl);
    return result_queue;
  }

  /// Peform ANN search over the bottom layer using batch dco
  ResultQueue _ann_search_level0_batch8(tableint seed, dist_t seed_dist, size_t ef) const {
    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        result_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        search_queue;

    // Init search
    dist_t max_dist = seed_dist;
    result_queue.emplace(max_dist, seed);
    search_queue.emplace(-max_dist, seed);
    visited[seed] = visited_array_tag;

    Id8 batched_nodes;
    Dist8 distances;
    Bool8 lt_flags{0};
    int n_batched = 0;

    auto process_remaining_batched_nodes = [&]() {
      for (int i = 0; i < n_batched; i++) {
        if (_dco.dist_comp(max_dist, batched_nodes[i], distances[i]) || result_queue.size() < ef) {
          search_queue.emplace(-distances[i], batched_nodes[i]);

          // Prefetch neighbor list of the top object in candidate queue
          prefetch_l1(_data_level0_memory + search_queue.top().second * _size_data_per_element +
                      _offset_level0);

          result_queue.emplace(distances[i], batched_nodes[i]);

          while (result_queue.size() > ef) {
            result_queue.pop();
          }

          if (!result_queue.empty()) max_dist = result_queue.top().first;
        }
      }
    };

    // Unbounded beam search
    while (true) {
      if (UNLIKELY(search_queue.empty())) {
        // No more nodes to visit.
        if (n_batched == 0) {
          // No batched nodes left; exit the loop.
          break;
        }

        // Process remaining batched nodes.
        process_remaining_batched_nodes();
        n_batched = 0;
        continue;
      }

      std::pair<dist_t, tableint> current_node_pair = search_queue.top();
      dist_t current_dist = -current_node_pair.first;
      search_queue.pop();

      if (UNLIKELY(current_dist > max_dist)) {
        // No more nodes to visit.
        if (n_batched == 0) {
          // No batched nodes left; exit the loop.
          break;
        }

        // Process remaining batched nodes.
        process_remaining_batched_nodes();
        n_batched = 0;
        continue;
      }

      tableint current_node_id = current_node_pair.second;
      int* neighbors = (int*)get_linklist0(current_node_id);
      size_t size = get_list_count((linklistsizeint*)neighbors);

      // Prefetch visited array
      prefetch_l1((char*)(visited + *(neighbors + 1)));
      prefetch_l1((char*)(visited + *(neighbors + 1) + 64));
      // Prefetch the second neighbor
      prefetch_l1((char*)(neighbors + 2));

      for (size_t j = 1; j <= size; j++) {
        // Prefetch visited array of the next neighbor
        prefetch_l1((char*)(visited + *(neighbors + j + 1)));
        // // Prefetch data of the next neighbor
        // _dco.prefetch(*(neighbors + j + 1));

        tableint neighbor_id = *(neighbors + j);

        if (!(visited[neighbor_id] == visited_array_tag)) {
          // Prefetch data for the first four batched neighbors
          if (n_batched > 0 && n_batched <= 5) {
            _dco.prefetch(batched_nodes[n_batched - 1]);
          }

          visited[neighbor_id] = visited_array_tag;

          if (UNLIKELY(result_queue.size() < ef)) {
            dist_t dist = _dco.compute(neighbor_id);
            search_queue.emplace(-dist, neighbor_id);
            result_queue.emplace(dist, neighbor_id);
            max_dist = result_queue.top().first;
          } else {
            if (n_batched == 8) {
              _dco.dist_comp8(max_dist, batched_nodes, distances, lt_flags);
              if (LIKELY(lt_flags.has_true())) {
                for (int i = 0; i < 8; i++) {
                  if (lt_flags.get(i)) {
                    search_queue.emplace(-distances[i], batched_nodes[i]);
                    result_queue.emplace(distances[i], batched_nodes[i]);
                  }
                }

                // Prefetch neighbor list of the top object in candidate queue
                prefetch_l1(_data_level0_memory +
                            search_queue.top().second * _size_data_per_element + _offset_level0);

                while (result_queue.size() > ef) {
                  result_queue.pop();
                }

                if (!result_queue.empty()) max_dist = result_queue.top().first;
              }
              n_batched = 0;
            }

            n_batched += 1;
            batched_nodes[n_batched - 1] = neighbor_id;
          }
        }
      }
    }
    _visited_list_pool->releaseVisitedList(vl);
    return result_queue;
  }

  /// Peform ANN search over the bottom layer using batch dco
  ResultQueue _ann_search_level0_batch8_with_refine_queue(tableint seed, dist_t seed_dist,
                                                          size_t refine_queue_size,
                                                          size_t ef) const {
    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        refine_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        cand_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        search_queue;

    // Init search
    dist_t max_result_dist = seed_dist;
    dist_t max_cand_dist = seed_dist;
    refine_queue.emplace(max_result_dist, seed);
    cand_queue.emplace(max_cand_dist, seed);
    search_queue.emplace(-max_cand_dist, seed);
    visited[seed] = visited_array_tag;

    Id8 batched_nodes;
    Dist8 lowerbounds;
    int n_batched = 0;

    auto push_cand_queue = [&](dist_t lowerbound, idx_t node_id) {
      search_queue.emplace(-lowerbound, node_id);

      // Prefetch neighbor list of the top object in search queue
      prefetch_l1(_data_level0_memory + search_queue.top().second * _size_data_per_element +
                  _offset_level0);

      if (lowerbound < max_result_dist || UNLIKELY(refine_queue.size() < refine_queue_size)) {
        dist_t distance = _dco.compute(node_id);
        refine_queue.emplace(distance, node_id);
        cand_queue.emplace(distance, node_id);
      } else {
        cand_queue.emplace(lowerbound, node_id);
      }

      while (refine_queue.size() > refine_queue_size) {
        refine_queue.pop();
      }
      while (cand_queue.size() > ef) {
        cand_queue.pop();
      }

      if (!refine_queue.empty()) max_result_dist = refine_queue.top().first;
      if (!cand_queue.empty()) max_cand_dist = cand_queue.top().first;
    };

    auto process_remaining_batched_nodes = [&]() {
      for (int i = 0; i < n_batched; i++) {
        dist_t lowerbound = _dco.relaxed_lowerbound(batched_nodes[i]);
        if (lowerbound < max_cand_dist || UNLIKELY(cand_queue.size() < ef)) {
          push_cand_queue(lowerbound, batched_nodes[i]);
        }
      }
    };

    // Unbounded beam search
    while (true) {
      if (UNLIKELY(search_queue.empty())) {
        // No more nodes to visit.
        if (n_batched == 0) {
          // No batched nodes left; exit the loop.
          break;
        }

        // Process remaining batched nodes.
        process_remaining_batched_nodes();
        n_batched = 0;
        continue;
      }

      std::pair<dist_t, tableint> current_node_pair = search_queue.top();
      dist_t current_dist = -current_node_pair.first;
      search_queue.pop();

      if (UNLIKELY(current_dist > max_cand_dist)) {
        // No more nodes to visit.
        if (n_batched == 0) {
          // No batched nodes left; exit the loop.
          break;
        }

        // Process remaining batched nodes.
        process_remaining_batched_nodes();
        n_batched = 0;
        continue;
      }

      tableint current_node_id = current_node_pair.second;
      int* neighbors = (int*)get_linklist0(current_node_id);
      size_t size = get_list_count((linklistsizeint*)neighbors);

      // Prefetch visited array
      prefetch_l1((char*)(visited + *(neighbors + 1)));
      prefetch_l1((char*)(visited + *(neighbors + 1) + 64));
      // Prefetch the second neighbor
      prefetch_l1((char*)(neighbors + 2));

      for (size_t j = 1; j <= size; j++) {
        // Prefetch visited array of the next neighbor
        prefetch_l1((char*)(visited + *(neighbors + j + 1)));

        tableint neighbor_id = *(neighbors + j);

        if (!(visited[neighbor_id] == visited_array_tag)) {
          // Prefetch data for the first four batched neighbors
          if (n_batched > 0 && n_batched <= 5) {
            _dco.prefetch(batched_nodes[n_batched - 1]);
          }

          visited[neighbor_id] = visited_array_tag;

          if (UNLIKELY(cand_queue.size() < ef)) {
            dist_t lowerbound = _dco.relaxed_lowerbound(neighbor_id);
            push_cand_queue(lowerbound, neighbor_id);
          } else {
            if (n_batched == 8) {
              _dco.relaxed_lowerbound8(batched_nodes, lowerbounds);

              for (int i = 0; i < 8; i++) {
                if (lowerbounds[i] < max_cand_dist) {
                  push_cand_queue(lowerbounds[i], batched_nodes[i]);
                }
              }

              n_batched = 0;
            }

            batched_nodes[n_batched] = neighbor_id;
            n_batched += 1;
          }
        }
      }
    }
    _visited_list_pool->releaseVisitedList(vl);
    return refine_queue;
  }

  linklistsizeint* get_linklist0(tableint internal_id) const {
    return (linklistsizeint*)(_data_level0_memory + internal_id * _size_data_per_element +
                              _offset_level0);
  }

  unsigned short int get_list_count(linklistsizeint* ptr) const {
    return *((unsigned short int*)ptr);
  }
};

}  // namespace detail
}  // namespace unity