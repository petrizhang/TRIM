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

#include "trim/common/common.h"
#include "trim/common/prefetch.h"
#include "trim/common/searcher.h"
#include "trim/common/setter_proxy.h"
#include "trim/detail/index/hnsw.h"
#include "trim/detail/searcher/hnsw/dco_exact.h"
#include "trim/detail/searcher/hnsw/dco_trim.h"

namespace trim {
namespace detail {

template <typename TDco = TrimDCO8<false>>
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
  std::shared_ptr<TrimHNSW> _shared_uhnsw{nullptr};  // a TrimHNSW index with shared ownership
  const TrimHNSW* _uhnsw{nullptr};                   // the underlying pointer of [shared_uhnsw]

  // Parameters
  size_t _ef{32};
  bool _enable_profile{false};
  bool _enable_batch_dco{true};
  mutable float _pruning_ratio{0.0f};
  mutable float _actual_distance_computation{0.0f};
  mutable float _total_distance_computation{0.0f};

  U_FORBID_COPY_AND_ASSIGN(HNSWSearcher);

  U_FORBID_MOVE(HNSWSearcher);

  HNSWSearcher() = delete;

  explicit HNSWSearcher(const std::shared_ptr<TrimHNSW>& index, TDco dco)
      : Proxy("HNSWSearcher"), _dco(std::move(dco)), _shared_uhnsw(index), _uhnsw(index.get()) {
    T_ASSERT(_uhnsw != nullptr);
    _hnsw = _uhnsw->owned_index_hnsw.get();
    T_ASSERT(_hnsw != nullptr);
    _visited_list_pool = _hnsw->visited_list_pool_.get();
    _data_level0_memory = _hnsw->data_level0_memory_;
    _size_data_per_element = _hnsw->size_data_per_element_;
    _size_links_per_element = _hnsw->size_links_per_element_;
    _offset_data = _hnsw->offsetData_;
    _offset_level0 = _hnsw->offsetLevel0_;
    _pruning_ratio = 0.0f;
    _actual_distance_computation = 0.0f;
    _total_distance_computation = 0.0f;

    Proxy::template bind<INTEGER_TYPE>("ef", &This::set_ef);
    Proxy::template bind<BOOL_TYPE>("enable_profile", &This::set_enable_profile);
    Proxy::template bind<BOOL_TYPE>("enable_batch_dco", &This::set_enable_batch_dco);
    Proxy::template bind<DOUBLE_TYPE>("gamma", &This::set_gamma);
  }

  ~HNSWSearcher() override = default;

  void set_data(float* data) override {};

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

  void range_search(const float* q, float radius, std::vector<int> &result) const override {_range_search(q, radius, result);}

  void optimize(int num_threads) override { T_THROW_MSG("not implemented error"); }

  IDCO* get_dco() const override { return &_dco; }

  Dict get_profile() const override { return _dco.get_profile(); }
  
  float get_pruning_ratio() const override { return _pruning_ratio; }
  
  float get_actual_distance_computation() const override { return _actual_distance_computation; }

  float get_total_distance_computation() const override { return _total_distance_computation; }
  
  void clear_pruning_ratio() const override { _pruning_ratio = 0.0; }

  void clear_num_distance_computation() const override { _actual_distance_computation = 0.0; _total_distance_computation = 0.0; }

  void set_ef(size_t ef) { _ef = ef; }

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
            T_THROW_FMT("search error, got illegal candidcate %u", cand);
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
    T_ASSERT(k > 0);
    const HnswlibIndex& index = *_hnsw;
    if (index.cur_element_count.load() == 0) return;

    _dco.set_query((dist_t*)query_data);
    dist_t seed_dist = std::numeric_limits<dist_t>::max();
    tableint seed = _init_search_seed(query_data, &seed_dist);

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        HnswlibIndex::CompareByFirst>
        top_candidates;

    if (_enable_batch_dco){
      if(_dco.get_gamma() == 0.0)
        top_candidates = _ann_search_level0_batch8_Gamma0(seed, seed_dist, k, std::max(_ef, k));
      else
        top_candidates = _ann_search_level0_batch8(seed, seed_dist, k, std::max(_ef, k));
    }
    else 
      top_candidates = _ann_search_level0(seed, seed_dist, k, std::max(_ef, k));

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
  ResultQueue _ann_search_level0(tableint seed, dist_t seed_dist, size_t result_set_size, size_t ef) const {
    
    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> result_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> cand_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> search_queue;

    // Init search
    dist_t max_result_dist = seed_dist;
    dist_t max_can_dist = seed_dist;
    result_queue.emplace(seed_dist, seed);
    cand_queue.emplace(seed_dist, seed);
    search_queue.emplace(-seed_dist, seed);
    visited[seed] = visited_array_tag;

    float total_data_access_count = 1.0;
    float actual_data_access_count = 1.0;

    // Unbounded beam search
    while (!search_queue.empty()) {
      
      std::pair<dist_t, tableint> current_node_pair = search_queue.top();
      dist_t current_dist = -current_node_pair.first;
      
      if (UNLIKELY(current_dist > max_can_dist) && cand_queue.size() == ef) {
        break;
      }

      search_queue.pop();

      tableint current_node_id = current_node_pair.second;
      int* neighbors = (int*) get_linklist0(current_node_id);
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

          total_data_access_count++;
          
          visited[neighbor_id] = visited_array_tag;

          if (UNLIKELY(cand_queue.size() < ef)) {

            actual_data_access_count++;

            dist_t dist = _dco.compute(neighbor_id);
            dist_t lowerbound = _dco.relaxed_lowerbound(neighbor_id);

            result_queue.emplace(dist, neighbor_id);
            cand_queue.emplace(dist, neighbor_id);
            search_queue.emplace(-lowerbound, neighbor_id);

            // Prefetch neighbor list of the top object in candidate queue
            prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0);
            prefetch_l1(_data_level0_memory + result_queue.top().second * _size_data_per_element + _offset_level0);
            
            while (result_queue.size() > result_set_size) result_queue.pop();
            if (!result_queue.empty()) max_result_dist = result_queue.top().first;
            if (!cand_queue.empty()) max_can_dist = cand_queue.top().first;

          } else {
            
            dist_t lowerbound = _dco.relaxed_lowerbound(neighbor_id);

            if (lowerbound < max_result_dist) {

              actual_data_access_count++;
              
              dist_t dist = _dco.compute(neighbor_id);
              
              result_queue.emplace(dist, neighbor_id);
              cand_queue.emplace(dist, neighbor_id);
              search_queue.emplace(-lowerbound, neighbor_id);

              // Prefetch neighbor list of the top object in candidate queue
              prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0);
              prefetch_l1(_data_level0_memory + result_queue.top().second * _size_data_per_element + _offset_level0);

              while (result_queue.size() > result_set_size) result_queue.pop();
              if (!result_queue.empty()) max_result_dist = result_queue.top().first;
              
              while (cand_queue.size() > ef) cand_queue.pop();
              if (!cand_queue.empty()) max_can_dist = cand_queue.top().first;

            } else {
              if (lowerbound < max_can_dist){
                
                cand_queue.emplace(lowerbound, neighbor_id);
                search_queue.emplace(-lowerbound, neighbor_id);

                prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0);

                while (cand_queue.size() > ef) cand_queue.pop();
                if (!cand_queue.empty()) max_can_dist = cand_queue.top().first;

              }
            }
          }
        }
      }
    }


    _actual_distance_computation += actual_data_access_count;
    _total_distance_computation += total_data_access_count;
    _pruning_ratio += (1 - actual_data_access_count/total_data_access_count);
    _visited_list_pool->releaseVisitedList(vl);

    return result_queue;
  }

  /// Peform ANN search over the bottom layer using batch dco
  ResultQueue _ann_search_level0_batch8(tableint seed, dist_t seed_dist,
                                                          size_t result_set_size,
                                                          size_t ef) const {
    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> result_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> cand_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> search_queue;

    // Init search
    dist_t max_result_dist = seed_dist;
    dist_t max_cand_dist = seed_dist;
    result_queue.emplace(max_result_dist, seed);
    cand_queue.emplace(max_cand_dist, seed);
    search_queue.emplace(-max_cand_dist, seed);
    visited[seed] = visited_array_tag;

    Id8 batched_nodes;
    Dist8 lowerbounds;
    int n_batched = 0;
    float total_data_access_count = 1.0;
    float actual_data_access_count = 1.0;

    auto push_cand_queue = [&](dist_t lowerbound, idx_t node_id) {
      
      search_queue.emplace(-lowerbound, node_id);

      if (lowerbound < max_result_dist || UNLIKELY(result_queue.size() < result_set_size)) {
        
        actual_data_access_count++;

        dist_t distance = _dco.compute(node_id);
        result_queue.emplace(distance, node_id);
        cand_queue.emplace(distance, node_id);

        prefetch_l1(_data_level0_memory + result_queue.top().second * _size_data_per_element + _offset_level0);
        prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0); 

        while (result_queue.size() > result_set_size) result_queue.pop();
        if (!result_queue.empty()) max_result_dist = result_queue.top().first;
        
        while (cand_queue.size() > ef) cand_queue.pop();
        if (!cand_queue.empty()) max_cand_dist = cand_queue.top().first;

      } else {

        cand_queue.emplace(lowerbound, node_id);

        prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0); 

        while (cand_queue.size() > ef) cand_queue.pop();
        if (!cand_queue.empty()) max_cand_dist = cand_queue.top().first;
      }   
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

      if (UNLIKELY(current_dist > max_cand_dist) && cand_queue.size() == ef) {
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
      int* neighbors = (int*) get_linklist0(current_node_id);
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

          total_data_access_count++;

          // Prefetch data for the first four batched neighbors
          if (n_batched > 0 && n_batched <= 5) {
            _dco.prefetch(batched_nodes[n_batched - 1]);
          }

          visited[neighbor_id] = visited_array_tag;

          if (UNLIKELY(cand_queue.size() < ef)) {
          // if (UNLIKELY(result_queue.size() < result_set_size)) {

            actual_data_access_count++;

            dist_t dist = _dco.compute(neighbor_id);
            dist_t lowerbound = _dco.relaxed_lowerbound(neighbor_id);

            result_queue.emplace(dist, neighbor_id);
            cand_queue.emplace(dist, neighbor_id);
            search_queue.emplace(-lowerbound, neighbor_id);

            // Prefetch neighbor list of the top object in candidate queue
            prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0);
            prefetch_l1(_data_level0_memory + result_queue.top().second * _size_data_per_element + _offset_level0);
            
            while (result_queue.size() > result_set_size) result_queue.pop();
            if (!result_queue.empty()) max_result_dist = result_queue.top().first;
            if (!cand_queue.empty()) max_cand_dist = cand_queue.top().first;

          } else {
            if (n_batched == 8) {
              
              _dco.relaxed_lowerbound8(batched_nodes, lowerbounds);

              for (int i = 0; i < 8; i++) {
                if (lowerbounds[i] < max_cand_dist || UNLIKELY(cand_queue.size() < ef)) {
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

    _actual_distance_computation += actual_data_access_count;
    _total_distance_computation += total_data_access_count;
    _pruning_ratio += (1 - actual_data_access_count/total_data_access_count);
    _visited_list_pool->releaseVisitedList(vl);
    return result_queue;
  }

  // Peform ANN search over the bottom layer with amma = 0 (using the original hnsw search algorithm)
  ResultQueue _ann_search_level0_batch8_Gamma0(tableint seed, dist_t seed_dist,
                                                          size_t result_set_size,
                                                          size_t ef) const {
    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    // Init search
    dist_t max_dist = seed_dist;
    top_candidates.emplace(max_dist, seed);
    candidateSet.emplace(-max_dist, seed);
    visited[seed] = visited_array_tag;

    int n_batched = 0;
    float total_data_access_count = 1.0;
    float actual_data_access_count = 1.0;

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
        if ((-curr_el_pair.first) > max_dist && top_candidates.size() == ef) {
            break;
        }
        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        int *data = (int*)get_linklist0(curNodeNum);
        size_t size = get_list_count((linklistsizeint*)data);
        tableint *datal = (tableint *) (data + 1);

#ifdef USE_SSE
        _mm_prefetch((char *) (visited + *(data + 1)), _MM_HINT_T0);
        _mm_prefetch((char *) (visited + *(data + 1) + 64), _MM_HINT_T0);
        // _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
        // _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

        for (size_t j = 0; j < size; j++) {
            tableint candidate_id = *(datal + j);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited + *(datal + j + 1)), _MM_HINT_T0);
            // _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
            if (visited[candidate_id] == visited_array_tag) continue;

            total_data_access_count++;
            visited[candidate_id] = visited_array_tag;

            dist_t lowerbound = _dco.relaxed_lowerbound(candidate_id);
            if (lowerbound < max_dist || UNLIKELY(top_candidates.size() < ef)) {
              
              actual_data_access_count++;
              
              dist_t dist1 = _dco.compute(candidate_id);
              candidateSet.emplace(-dist1, candidate_id);
              top_candidates.emplace(dist1, candidate_id);
#ifdef USE_SSE
              prefetch_l1(_data_level0_memory + candidateSet.top().second * _size_data_per_element + _offset_level0);   
#endif
              if (top_candidates.size() > ef)
                  top_candidates.pop();

              if (!top_candidates.empty())
                  max_dist = top_candidates.top().first;   
            }
          }
        }
        
       
    _actual_distance_computation += actual_data_access_count;
    _total_distance_computation += total_data_access_count;
    _pruning_ratio += (1 - actual_data_access_count/total_data_access_count);        
    _visited_list_pool->releaseVisitedList(vl);

    return top_candidates;


    // VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    // vl_type* visited = vl->mass;
    // vl_type visited_array_tag = vl->curV;

    // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> cand_queue;
    // std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> search_queue;

    // // Init search
    // dist_t max_cand_dist = seed_dist;
    // cand_queue.emplace(max_cand_dist, seed);
    // search_queue.emplace(-max_cand_dist, seed);
    // visited[seed] = visited_array_tag;

    // Id8 batched_nodes;
    // Dist8 lowerbounds;
    // int n_batched = 0;
    // float total_data_access_count = 1.0;
    // float actual_data_access_count = 1.0;

    // auto push_cand_queue = [&](dist_t lowerbound, idx_t node_id) {

    //   actual_data_access_count++;
      
    //   dist_t dist = _dco.compute(node_id);
    //   cand_queue.emplace(dist, node_id);
    //   search_queue.emplace(-dist, node_id);

    //   prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0); 

    //   while (cand_queue.size() > ef) cand_queue.pop();
    //   if (!cand_queue.empty()) max_cand_dist = cand_queue.top().first;

    // };

    // auto process_remaining_batched_nodes = [&]() {
    //   for (int i = 0; i < n_batched; i++) {
    //     dist_t lowerbound = _dco.relaxed_lowerbound(batched_nodes[i]);
    //     if (lowerbound < max_cand_dist || UNLIKELY(cand_queue.size() < ef)) {
    //       push_cand_queue(lowerbound, batched_nodes[i]);
    //     }
    //   }
    // };

    // // Unbounded beam search
    // while (true) {
      
    //   if (UNLIKELY(search_queue.empty())) {
    //     // No more nodes to visit.
    //     if (n_batched == 0) {
    //       // No batched nodes left; exit the loop.
    //       break;
    //     }

    //     // Process remaining batched nodes.
    //     process_remaining_batched_nodes();
    //     n_batched = 0;
    //     continue;
    //   }

    //   std::pair<dist_t, tableint> current_node_pair = search_queue.top();
    //   dist_t current_dist = -current_node_pair.first;
    //   search_queue.pop();

    //   if (UNLIKELY(current_dist > max_cand_dist) && cand_queue.size() == ef) {
    //     // No more nodes to visit.
    //     if (n_batched == 0) {
    //       // No batched nodes left; exit the loop.
    //       break;
    //     }

    //     // Process remaining batched nodes.
    //     process_remaining_batched_nodes();
    //     n_batched = 0;
    //     continue;
    //   }

    //   tableint current_node_id = current_node_pair.second;
    //   int* neighbors = (int*) get_linklist0(current_node_id);
    //   size_t size = get_list_count((linklistsizeint*)neighbors);

    //   // Prefetch visited array
    //   prefetch_l1((char*)(visited + *(neighbors + 1)));
    //   prefetch_l1((char*)(visited + *(neighbors + 1) + 64));
    //   // Prefetch the second neighbor
    //   prefetch_l1((char*)(neighbors + 2));

    //   for (size_t j = 1; j <= size; j++) {
    //     // Prefetch visited array of the next neighbor
    //     prefetch_l1((char*)(visited + *(neighbors + j + 1)));

    //     tableint neighbor_id = *(neighbors + j);

    //     if (!(visited[neighbor_id] == visited_array_tag)) {

    //       total_data_access_count++;

    //       // Prefetch data for the first four batched neighbors
    //       if (n_batched > 0 && n_batched <= 5) {
    //         _dco.prefetch(batched_nodes[n_batched - 1]);
    //       }

    //       visited[neighbor_id] = visited_array_tag;

    //       if (UNLIKELY(cand_queue.size() < ef)) {

    //         actual_data_access_count++;

    //         dist_t dist = _dco.compute(neighbor_id);

    //         cand_queue.emplace(dist, neighbor_id);
    //         search_queue.emplace(-dist, neighbor_id);

    //         // Prefetch neighbor list of the top object in candidate queue
    //         prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0);
            
    //         if (!cand_queue.empty()) max_cand_dist = cand_queue.top().first;

    //       } else {
    //         if (n_batched == 8) {
              
    //           _dco.relaxed_lowerbound8(batched_nodes, lowerbounds);

    //           for (int i = 0; i < 8; i++) {
    //             if (lowerbounds[i] < max_cand_dist || UNLIKELY(cand_queue.size() < ef)) {
    //               push_cand_queue(lowerbounds[i], batched_nodes[i]);
    //             }
    //           }

    //           n_batched = 0;
    //         }

    //         batched_nodes[n_batched] = neighbor_id;
    //         n_batched += 1;
    //       }
    //     }
    //   }
    // }

    // _actual_distance_computation += actual_data_access_count;
    // _total_distance_computation += total_data_access_count;
    // _pruning_ratio += (1 - actual_data_access_count/total_data_access_count);
    // _visited_list_pool->releaseVisitedList(vl);
    // return cand_queue;
  }

  /// Range search implementation
  void _range_search(const void* query_data, dist_t dist, std::vector<int> &result) const {

    const HnswlibIndex& index = *_hnsw;
    if (index.cur_element_count.load() == 0) return;

    _dco.set_query((dist_t*)query_data);
    dist_t seed_dist = std::numeric_limits<dist_t>::max();
    tableint seed = _init_search_seed(query_data, &seed_dist);

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        HnswlibIndex::CompareByFirst>
        top_candidates;

    if (_enable_batch_dco){
      if(_dco.get_gamma() == 0.0)
        top_candidates = _range_search_level0_batch8_Gamma0(seed, seed_dist, dist, _ef);
      else
        top_candidates = _range_search_level0_batch8(seed, seed_dist, dist, _ef);
    }
    else 
      top_candidates = _range_search_level0(seed, seed_dist, dist, _ef);

    while (!top_candidates.empty()) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      result.push_back(index.getExternalLabel(rez.second));
      top_candidates.pop();
    }

    return;

  }

  /// Peform Range search over the bottom layer
  ResultQueue _range_search_level0(tableint seed, dist_t seed_dist, dist_t dist, size_t ef) const {

    dist_t dist2 = dist*dist;
    
    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> result_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> cand_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> search_queue;

    // Init search
    dist_t max_can_dist = seed_dist;
    if (seed_dist <= dist2) result_queue.emplace(seed_dist, seed);
    cand_queue.emplace(seed_dist, seed);
    search_queue.emplace(-seed_dist, seed);
    visited[seed] = visited_array_tag;

    float total_data_access_count = 1.0;
    float actual_data_access_count = 1.0;

    // Unbounded beam search
    while (!search_queue.empty()) {
      
      std::pair<dist_t, tableint> current_node_pair = search_queue.top();
      dist_t current_dist = -current_node_pair.first;
      
      if (UNLIKELY(current_dist > max_can_dist) && cand_queue.size() == ef) {
        break;
      }

      search_queue.pop();

      tableint current_node_id = current_node_pair.second;
      int* neighbors = (int*) get_linklist0(current_node_id);
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

          total_data_access_count++;
          
          visited[neighbor_id] = visited_array_tag;

          if (UNLIKELY(cand_queue.size() < ef)) {

            actual_data_access_count++;

            dist_t dist = _dco.compute(neighbor_id);
            dist_t lowerbound = _dco.relaxed_lowerbound(neighbor_id);

            if(dist <= dist2) result_queue.emplace(dist, neighbor_id);
            cand_queue.emplace(dist, neighbor_id);
            search_queue.emplace(-lowerbound, neighbor_id);

            // Prefetch neighbor list of the top object in candidate queue
            prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0);

            if (!cand_queue.empty()) max_can_dist = cand_queue.top().first;

          } else {
            
            dist_t lowerbound = _dco.relaxed_lowerbound(neighbor_id);

            if (lowerbound <= dist2) {

              actual_data_access_count++;
              
              dist_t dist = _dco.compute(neighbor_id);
              
              if(dist <= dist2) result_queue.emplace(dist, neighbor_id);
              cand_queue.emplace(dist, neighbor_id);
              search_queue.emplace(-lowerbound, neighbor_id);

              // Prefetch neighbor list of the top object in candidate queue
              prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0);
              
              while (cand_queue.size() > ef) cand_queue.pop();
              if (!cand_queue.empty()) max_can_dist = cand_queue.top().first;

            } else {
              if (lowerbound < max_can_dist){
                
                cand_queue.emplace(lowerbound, neighbor_id);
                search_queue.emplace(-lowerbound, neighbor_id);

                prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0);

                while (cand_queue.size() > ef) cand_queue.pop();
                if (!cand_queue.empty()) max_can_dist = cand_queue.top().first;

              }
            }
          }
        }
      }
    }

    _actual_distance_computation += actual_data_access_count;
    _total_distance_computation += total_data_access_count;
    _pruning_ratio += (1 - actual_data_access_count/total_data_access_count);
    _visited_list_pool->releaseVisitedList(vl);

    return result_queue;

  }

  /// Peform Range search over the bottom layer using batch dco
  ResultQueue _range_search_level0_batch8(tableint seed, dist_t seed_dist, dist_t dist, size_t ef) const {
    
    dist_t dist2 = dist * dist;

    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> result_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> cand_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> search_queue;

    // Init search
    dist_t max_cand_dist = seed_dist;
    if (seed_dist <= dist2) result_queue.emplace(seed_dist, seed);
    cand_queue.emplace(seed_dist, seed);
    search_queue.emplace(-seed_dist, seed);
    visited[seed] = visited_array_tag;

    Id8 batched_nodes;
    Dist8 lowerbounds;
    int n_batched = 0;
    float total_data_access_count = 1.0;
    float actual_data_access_count = 1.0;

    auto push_cand_queue = [&](dist_t lowerbound, idx_t node_id) {
      
      search_queue.emplace(-lowerbound, node_id);

      if (lowerbound <= dist2) {
        
        actual_data_access_count++;

        dist_t distance = _dco.compute(node_id);
        if (distance <= dist2) result_queue.emplace(distance, node_id);
        cand_queue.emplace(distance, node_id);

        prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0); 
        
        while (cand_queue.size() > ef) cand_queue.pop();
        if (!cand_queue.empty()) max_cand_dist = cand_queue.top().first;

      } else {

        cand_queue.emplace(lowerbound, node_id);

        prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0); 

        while (cand_queue.size() > ef) cand_queue.pop();
        if (!cand_queue.empty()) max_cand_dist = cand_queue.top().first;
      }   
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

      if (UNLIKELY(current_dist > max_cand_dist) && cand_queue.size() == ef) {
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
      int* neighbors = (int*) get_linklist0(current_node_id);
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

          total_data_access_count++;

          // Prefetch data for the first four batched neighbors
          if (n_batched > 0 && n_batched <= 5) {
            _dco.prefetch(batched_nodes[n_batched - 1]);
          }

          visited[neighbor_id] = visited_array_tag;

          if (UNLIKELY(cand_queue.size() < ef)) {

            actual_data_access_count++;

            dist_t dist = _dco.compute(neighbor_id);
            dist_t lowerbound = _dco.relaxed_lowerbound(neighbor_id);

            if (dist <= dist2) result_queue.emplace(dist, neighbor_id);
            cand_queue.emplace(dist, neighbor_id);
            search_queue.emplace(-lowerbound, neighbor_id);

            // Prefetch neighbor list of the top object in candidate queue
            prefetch_l1(_data_level0_memory + cand_queue.top().second * _size_data_per_element + _offset_level0);
            
            if (!cand_queue.empty()) max_cand_dist = cand_queue.top().first;

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

    _actual_distance_computation += actual_data_access_count;
    _total_distance_computation += total_data_access_count;
    _pruning_ratio += (1 - actual_data_access_count/total_data_access_count);
    _visited_list_pool->releaseVisitedList(vl);
    return result_queue;
  }
  
  // Peform Rang search over the bottom layer with amma = 0 (using the original hnsw search algorithm)
  ResultQueue _range_search_level0_batch8_Gamma0(tableint seed, dist_t seed_dist, dist_t dist, size_t ef) const {
    
    dist_t dist2 = dist * dist;

    VisitedList* vl = _visited_list_pool->getFreeVisitedList();
    vl_type* visited = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> result_queue;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

    // Init search
    dist_t max_dist = seed_dist;
    if(seed_dist <= dist2) result_queue.emplace(max_dist, seed);
    top_candidates.emplace(max_dist, seed);
    candidateSet.emplace(-max_dist, seed);
    visited[seed] = visited_array_tag;

    int n_batched = 0;
    float total_data_access_count = 1.0;
    float actual_data_access_count = 1.0;

    while (!candidateSet.empty()) {
    std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
    if ((-curr_el_pair.first) > max_dist && top_candidates.size() == ef) {
    break;
    }
    candidateSet.pop();

    tableint curNodeNum = curr_el_pair.second;

    int *data = (int*)get_linklist0(curNodeNum);
    size_t size = get_list_count((linklistsizeint*)data);
    tableint *datal = (tableint *) (data + 1);

    #ifdef USE_SSE
    _mm_prefetch((char *) (visited + *(data + 1)), _MM_HINT_T0);
    _mm_prefetch((char *) (visited + *(data + 1) + 64), _MM_HINT_T0);
    // _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
    // _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
    #endif

    for (size_t j = 0; j < size; j++) {
      
        tableint candidate_id = *(datal + j);

        #ifdef USE_SSE
       _mm_prefetch((char *) (visited + *(datal + j + 1)), _MM_HINT_T0);
        // _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
        #endif

        if (visited[candidate_id] == visited_array_tag) continue;

        total_data_access_count++;
        visited[candidate_id] = visited_array_tag;

        dist_t lowerbound = _dco.relaxed_lowerbound(candidate_id);

        if (lowerbound <= dist2 || lowerbound < max_dist || UNLIKELY(top_candidates.size() < ef)) {

          actual_data_access_count++;

          dist_t dist1 = _dco.compute(candidate_id);
          if(lowerbound <= dist2) 
            result_queue.emplace(dist1, candidate_id);     
          candidateSet.emplace(-dist1, candidate_id);
          top_candidates.emplace(dist1, candidate_id);
          
          #ifdef USE_SSE
          prefetch_l1(_data_level0_memory + candidateSet.top().second * _size_data_per_element + _offset_level0);   
          #endif

          if (top_candidates.size() > ef)
            top_candidates.pop();

          if (!top_candidates.empty())
            max_dist = top_candidates.top().first;   
        }
      }
    }

    _actual_distance_computation += actual_data_access_count;
    _total_distance_computation += total_data_access_count;
    _pruning_ratio += (1 - actual_data_access_count/total_data_access_count);        
    _visited_list_pool->releaseVisitedList(vl);

    return result_queue;
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
}  // namespace trim