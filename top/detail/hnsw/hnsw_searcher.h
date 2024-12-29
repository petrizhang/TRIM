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

#include "top/common/searcher.h"
#include "top/detail/hnswlib/hnswlib.h"

namespace top {
namespace detail {
using Space = hnswlib::SpaceInterface<float>;
using HnswlibIndex = hnswlib::HierarchicalNSW<float>;
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

struct HNSWSearcher : Searcher {
  std::unique_ptr<Space> owned_space;
  std::unique_ptr<HnswlibIndex> owned_index;
  bool use_hnswlib = false;
  size_t ef = 32;

  HNSWSearcher() = default;
  ~HNSWSearcher() override = default;
  void set_data(const float* data, int n, int dim) override;
  void ann_search(const float* q, int k, int* dst) const override;
  void range_search(const float* q, float radius, int* dst) const override;
  void set(const std::string& key, Object value) override;
  void optimize(int num_threads) override;
  Dict get_profile() const override;

  void _index_ann_search(const void* query_data, size_t k, int* dst) const;
  ResultQueue _index_ann_search_base_layer(tableint ep_id, const void* data_point, size_t ef) const;
};

inline void HNSWSearcher::set_data(const float* data, int n, int dim) { return; }

inline void HNSWSearcher::ann_search(const float* q, int k, int* dst) const {
  if (use_hnswlib) {
    owned_index->setEf(ef);
    auto ret = owned_index->searchKnn(q, k);
    {
      size_t sz = ret.size();
      while (!ret.empty()) {
        dst[--sz] = ret.top().second;
        ret.pop();
      }
    }
  } else {
    _index_ann_search(q, k, dst);
  }
}

inline void HNSWSearcher::range_search(const float* q, float radius, int* dst) const {
  TOP_THROW_MSG("not implemented error");
}

inline void HNSWSearcher::set(const std::string& key, Object value) {
  if (key == "ef") {
    TOP_THROW_IF_NOT_MSG(value.type == ObjectType::INTEGER_TYPE, "`ef` must be an integer");
    ef = static_cast<int>(value.get_integer());
  } else if (key == "use_hnswlib") {
    TOP_THROW_IF_NOT_MSG(value.type == ObjectType::BOOL_TYPE, "`ef` must be an integer");
    use_hnswlib = value.get_bool();
  } else {
    TOP_THROW_FMT("unknown parameter %s", key.c_str());
  }
}

inline void HNSWSearcher::optimize(int num_threads) { TOP_THROW_MSG("not implemented error"); }

inline Dict HNSWSearcher::get_profile() const { return {}; }

inline void HNSWSearcher::_index_ann_search(const void* query_data, size_t k, int* dst) const {
  const HnswlibIndex& index = *owned_index;
  auto& fstdistfunc_ = index.fstdistfunc_;
  if (index.cur_element_count == 0) return;

  tableint currObj = index.enterpoint_node_;
  dist_t curdist = fstdistfunc_(query_data, index.getDataByInternalId(index.enterpoint_node_),
                                index.dist_func_param_);

  for (int level = index.maxlevel_; level > 0; level--) {
    bool changed = true;
    while (changed) {
      changed = false;
      unsigned int* data;

      data = (unsigned int*)index.get_linklist(currObj, level);
      int size = index.getListCount(data);
      // metric_hops++;
      // metric_distance_computations += size;

      tableint* datal = (tableint*)(data + 1);
      for (int i = 0; i < size; i++) {
        tableint cand = datal[i];
        if (cand < 0 || cand > index.max_elements_) throw std::runtime_error("cand error");
        dist_t d =
            fstdistfunc_(query_data, index.getDataByInternalId(cand), index.dist_func_param_);

        if (d < curdist) {
          curdist = d;
          currObj = cand;
          changed = true;
        }
      }
    }
  }

  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                      HnswlibIndex::CompareByFirst>
      top_candidates;

  top_candidates = _index_ann_search_base_layer(currObj, query_data, std::max(ef, k));

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
}

inline ResultQueue HNSWSearcher::_index_ann_search_base_layer(tableint ep_id,
                                                              const void* data_point,
                                                              size_t ef) const {
  const HnswlibIndex& index = *owned_index;
  auto* visited_list_pool_ = index.visited_list_pool_.get();
  auto& fstdistfunc_ = index.fstdistfunc_;
  auto* dist_func_param_ = index.dist_func_param_;
  auto* data_level0_memory_ = index.data_level0_memory_;
  auto size_data_per_element_ = index.size_data_per_element_;
  auto offsetLevel0_ = index.offsetLevel0_;
  auto offsetData_ = index.offsetData_;

  VisitedList* vl = visited_list_pool_->getFreeVisitedList();
  vl_type* visited_array = vl->mass;
  vl_type visited_array_tag = vl->curV;

  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                      CompareByFirst>
      top_candidates;
  std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                      CompareByFirst>
      candidate_set;

  // Init search
  char* ep_data = index.getDataByInternalId(ep_id);
  dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
  dist_t lowerBound = dist;
  top_candidates.emplace(dist, ep_id);
  candidate_set.emplace(-dist, ep_id);
  visited_array[ep_id] = visited_array_tag;

  // Unbounded beam search
  while (!candidate_set.empty()) {
    std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
    dist_t candidate_dist = -current_node_pair.first;
    if (candidate_dist > lowerBound) {
      break;
    }
    candidate_set.pop();

    tableint current_node_id = current_node_pair.second;
    int* data = (int*)index.get_linklist0(current_node_id);
    size_t size = index.getListCount((linklistsizeint*)data);

#ifdef USE_SSE
    _mm_prefetch((char*)(visited_array + *(data + 1)), _MM_HINT_T0);
    _mm_prefetch((char*)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
    _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_,
                 _MM_HINT_T0);
    _mm_prefetch((char*)(data + 2), _MM_HINT_T0);
#endif

    for (size_t j = 1; j <= size; j++) {
      int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
      _mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0);
      _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                   _MM_HINT_T0);  ////////////
#endif
      if (!(visited_array[candidate_id] == visited_array_tag)) {
        visited_array[candidate_id] = visited_array_tag;

        char* currObj1 = (index.getDataByInternalId(candidate_id));
        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
        if (top_candidates.size() < ef || lowerBound > dist) {
          candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
          _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                           offsetLevel0_,  ///////////
                       _MM_HINT_T0);       ////////////////////////
#endif

          top_candidates.emplace(dist, candidate_id);

          while (top_candidates.size() > ef) {
            top_candidates.pop();
          }

          if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
        }
      }
    }
  }

  visited_list_pool_->releaseVisitedList(vl);
  return top_candidates;
}

}  // namespace detail
}  // namespace top