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
#include <atomic>
#include <memory>
#include <queue>

#include "unity/common/searcher.h"
#include "unity/detail/uhnsw/exact_dco.h"
#include "unity/detail/uhnsw/unity_hnsw.h"
#include "unity/detail/uhnsw/unity_op.h"

namespace unity {
namespace detail {

using CompareByFirst = HnswlibIndex::CompareByFirst;
using VisitedList = hnswlib::VisitedList;
using dist_t = float;
using labeltype = hnswlib::labeltype;
using tableint = hnswlib::tableint;
using vl_type = hnswlib::vl_type;
using linklistsizeint = hnswlib::linklistsizeint;
using ResultQueue =
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        HnswlibIndex::CompareByFirst>;

template <typename DCO = UnityOp8<false>>
struct HNSWSearcher : Searcher {
  // DCO
  mutable DCO _dco;

  // HNSW members
  hnswlib::VisitedListPool* _visited_list_pool{nullptr};
  char* _data_level0_memory{nullptr};
  size_t _size_data_per_element{0};
  size_t _size_links_per_element{0};
  size_t _offset_data{0}, _offset_level0{0};

  // Index
  const HnswlibIndex* _hnsw{nullptr};
  std::shared_ptr<UnityHNSW> _shared_uhnsw{nullptr};  // a UnityHNSW index with shared ownership
  const UnityHNSW* _uhnsw{nullptr};                   // the underlying pointer of [shared_uhnsw]

  // Parameters
  size_t _ef{32};
  bool _enable_profile{false};

  HNSWSearcher() = delete;

  explicit HNSWSearcher(const std::shared_ptr<UnityHNSW>& index, DCO dco)
      : _dco(std::move(dco)), _shared_uhnsw(index), _uhnsw(index.get()) {
    U_ASSERT(_uhnsw != nullptr);
    _hnsw = _uhnsw->owned_index_hnsw.get();
    U_ASSERT(_hnsw != nullptr);
    _visited_list_pool = _hnsw->visited_list_pool_.get();
    _data_level0_memory = _hnsw->data_level0_memory_;
    _size_data_per_element = _hnsw->size_data_per_element_;
    _size_links_per_element = _hnsw->size_links_per_element_;
    _offset_data = _hnsw->offsetData_;
    _offset_level0 = _hnsw->offsetLevel0_;
  }

  ~HNSWSearcher() override = default;

  void set_data(const float* data, int n, int dim) override final {};

  void ann_search(const float* q, int k, int* dst) const override final { _ann_search(q, k, dst); };

  void range_search(const float* q, float radius, int* dst) const override final {
    U_THROW_MSG("not implemented error");
  };

  void set(const std::string& key, Object value) override {
    if (key == "ef") {
      U_THROW_IF_NOT_MSG(value.type == ObjectType::INTEGER_TYPE,
                         "parameter `ef` must be an integer");
      _ef = static_cast<int>(value.get_integer());
    } else if (key == "enable_profile") {
      U_THROW_IF_NOT_MSG(value.type == ObjectType::BOOL_TYPE,
                         "parameter  `enable_profile` must be a boolean value");
      _enable_profile = value.get_bool();
    } else {
      U_THROW_FMT("unknown parameter %s", key.c_str());
    }
  };

  void optimize(int num_threads) override final { U_THROW_MSG("not implemented error"); };

  Dict get_profile() const override final { return _dco.get_profile(); };

  inline tableint _init_search_seed(const void* query_data, dist_t* dist) const {
    const HnswlibIndex& index = *_uhnsw->owned_index_hnsw;

    tableint node = index.enterpoint_node_;
    dist_t curdist = -1;
    _dco.distance_less_than(-1, index.enterpoint_node_, &curdist);

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
          dist_t d = -1;
          if (_dco.distance_less_than(curdist, cand, &d)) {
            curdist = d;
            node = cand;
            changed = true;
          }
        }
      }
    }
    *dist = curdist;
    return node;
  };

  /// ANN search implementation
  inline void _ann_search(const void* query_data, size_t k, int* dst) const {
    const HnswlibIndex& index = *_uhnsw->owned_index_hnsw;
    if (index.cur_element_count.load() == 0) return;

    _dco.set_query((dist_t*)query_data);
    dist_t seed_dist = -1;
    tableint seed = _init_search_seed(query_data, &seed_dist);

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        HnswlibIndex::CompareByFirst>
        top_candidates = _ann_search_level0(seed, seed_dist, std::max(_ef, k));

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
  };

  /// Peform ANN search over the bottom layer
  inline ResultQueue _ann_search_level0(tableint seed, dist_t seed_dist, size_t ef) const {
    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        results;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidates;

    // Init search
    dist_t max_dist = seed_dist;
    results.emplace(max_dist, seed);
    candidates.emplace(-max_dist, seed);
    visited[seed] = visited_array_tag;

    // Unbounded beam search
    while (!candidates.empty()) {
      std::pair<dist_t, tableint> current_node_pair = candidates.top();
      dist_t candidate_dist = -current_node_pair.first;
      if (candidate_dist > max_dist) {
        break;
      }
      candidates.pop();

      tableint current_node_id = current_node_pair.second;
      int* neighbors = (int*)get_linklist0(current_node_id);
      size_t size = get_list_count((linklistsizeint*)neighbors);

      // Prefetch visited array
      _prefetch((char*)(visited + *(neighbors + 1)));
      _prefetch((char*)(visited + *(neighbors + 1) + 64));
      // Prefetch vector data of the first neighbor
      _prefetch(_data_level0_memory + (*(neighbors + 1)) * _size_data_per_element + _offset_data);
      // Prefetch the second neighbor
      _prefetch((char*)(neighbors + 2));

      for (size_t j = 1; j <= size; j++) {
        int cand_id = *(neighbors + j);

        // Prefetch visited array of the next neighbor
        _prefetch((char*)(visited + *(neighbors + j + 1)));
        // Prefetch vector data of the next neighbor
        _prefetch(_data_level0_memory + (*(neighbors + j + 1)) * _size_data_per_element +
                  _offset_data);

        if (!(visited[cand_id] == visited_array_tag)) {
          visited[cand_id] = visited_array_tag;

          dist_t dist = -1;
          bool found_better_cand = _dco.distance_less_than(max_dist, cand_id, &dist);
          if (found_better_cand || results.size() < ef) {
            candidates.emplace(-dist, cand_id);

            // Prefetch neighbor list of the top object in candidate queue
            _prefetch(_data_level0_memory + candidates.top().second * _size_data_per_element +
                      _offset_level0);

            results.emplace(dist, cand_id);

            while (results.size() > ef) {
              results.pop();
            }

            if (!results.empty()) max_dist = results.top().first;
          }
        }
      }
    }

    _visited_list_pool->releaseVisitedList(vl);
    return results;
  }

  void _prefetch(const void* p) const {
#ifdef USE_SSE
    _mm_prefetch(p, _MM_HINT_T0);
#endif
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