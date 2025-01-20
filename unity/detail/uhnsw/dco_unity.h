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

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#include "unity/common/atomic.h"
#include "unity/common/dco.h"
#include "unity/detail/faiss/impl/ProductQuantizer.h"
#include "unity/detail/faiss/impl/code_distance/code_distance.h"
#include "unity/detail/hnswlib/hnswlib.h"
#include "unity/detail/uhnsw/unity_hnsw.h"

namespace unity {
namespace detail {

template <typename PQDecoderType, bool enable_profile = false>
struct UnityOp final : IDistanceComparisonOperator<unsigned, float> {
  using idx_t = unsigned;
  using dist_t = float;

  // Data structures for query
  const dist_t* _dist_table_data{nullptr};
  const uint8_t* _codes;
  const float* _recons_errors;
  size_t _M{0};
  size_t _nbits{0};
  size_t _code_size{0};
  hnswlib::DISTFUNC<dist_t> _dist_func{nullptr};
  void* _dist_func_param{nullptr};
  AlignedTable<dist_t> _dist_table;
  const IndexPQ* _pq{nullptr};
  const hnswlib::HierarchicalNSW<float>* _hnsw{nullptr};
  const dist_t* _query{nullptr};

  // Profile metrics
  Atomic<int64_t> _num_distance_computation;
  Atomic<int64_t> _num_pq_distance_computation;
  Atomic<int64_t> _num_lowerbound_computation;

  // Search parameters
  float gamma{0.8};

  ~UnityOp() override = default;

  explicit UnityOp(const UnityHNSW* uhnsw) {
    U_ASSERT(uhnsw != nullptr);
    U_ASSERT(uhnsw->owned_index_hnsw != nullptr);
    U_ASSERT(uhnsw->owned_index_pq != nullptr);
    U_ASSERT(uhnsw->owned_space != nullptr);
    _pq = uhnsw->owned_index_pq.get();
    _M = _pq->quantizer.M;
    _nbits = _pq->quantizer.nbits;
    _code_size = _pq->code_size;
    _codes = _pq->codes.data();
    _recons_errors = _pq->recons_errors.data();

    _hnsw = uhnsw->owned_index_hnsw.get();
    _dist_func = _hnsw->fstdistfunc_;
    _dist_func_param = _hnsw->dist_func_param_;
  }

  void set_query(const dist_t* query_data) override {
    _query = query_data;
    _dist_table.resize(_pq->quantizer.M * _pq->quantizer.ksub);
    _pq->quantizer.compute_distance_table(_query, _dist_table.data());
    _dist_table_data = _dist_table.data();
  }

  bool distance_less_than(dist_t max_dist, idx_t i, float* dist) const override final {
    dist_t lowerbound = relaxed_lowerbound(i);
    if (lowerbound >= max_dist) {
      return false;
    }

    *dist = compute(i);
    return *dist < max_dist;
  }

  bool distance4_less_than(dist_t max_dist, idx_t i0, idx_t i1, idx_t i2, idx_t i3,
                           float* __restrict dist4, bool4& flag4) const override final {
    flag4.mask = 0;

    float a[4] = {0, 0, 0, 0};
    float b[4] = {0, 0, 0, 0};
    float lowerbounds[4] = {0, 0, 0, 0};

    _prefetch(_recons_errors + i0);
    _prefetch(_recons_errors + i1);
    _prefetch(_recons_errors + i2);
    _prefetch(_recons_errors + i3);

    faiss::distance_four_codes<PQDecoderType>(
        _M, _nbits, _dist_table_data, _codes + i0 * _code_size, _codes + i1 * _code_size,
        _codes + i2 * _code_size, _codes + i3 * _code_size, a[0], a[1], a[2], a[3]);

    b[0] = _recons_errors[i0];
    b[1] = _recons_errors[i1];
    b[2] = _recons_errors[i2];
    b[3] = _recons_errors[i3];

    for (int i = 0; i < 4; i++) {
      a[i] = std::sqrt(a[i]);
    }

    for (int i = 0; i < 4; i++) {
      lowerbounds[i] = (a[i] - b[i]) * (a[i] - b[i]) + 2 * gamma * a[i] * b[i];
    }

    if (lowerbounds[0] < max_dist) {
      dist4[0] = compute(i0);
      flag4.set_bool0(dist4[0] < max_dist);
    }

    if (lowerbounds[1] < max_dist) {
      dist4[1] = compute(i1);
      flag4.set_bool1(dist4[1] < max_dist);
    }

    if (lowerbounds[2] < max_dist) {
      dist4[2] = compute(i2);
      flag4.set_bool2(dist4[2] < max_dist);
    }

    if (lowerbounds[3] < max_dist) {
      dist4[3] = compute(i3);
      flag4.set_bool3(dist4[3] < max_dist);
    }

    return flag4.has_true();
  }

  dist_t compute(idx_t i) const override {
    assert(_query != nullptr);
    return _dist_func(_query, _hnsw->getDataByInternalId(i), _dist_func_param);
  }

  dist_t relaxed_lowerbound(idx_t i) const override {
    dist_t a = std::sqrt(estimate(i));
    dist_t b = _recons_errors[i];
    return (a - b) * (a - b) + 2 * gamma * a * b;
  }

  dist_t estimate(idx_t i) const override {
    return faiss::distance_single_code<PQDecoderType>(_M, _nbits, _dist_table_data,
                                                      _codes + i * _code_size);
  }

  void set(const std::string& key, const Object& value) override {
    if (key == "gamma") {
      U_THROW_IF_NOT_MSG(value.type == ObjectType::DOUBLE_TYPE,
                         "parameter `ef` must be a double value");
      gamma = static_cast<float>(value.get_double());
    } else {
      U_THROW_FMT("unknown parameter %s", key.c_str());
    }
  }

  void prefetch(idx_t i) const override {
#ifdef __SSE__
    _mm_prefetch(_recons_errors + i, _MM_HINT_T0);
    _mm_prefetch((char*)(_codes + _code_size * i), _MM_HINT_T0);
#endif
  }

  void _prefetch(const void* p) const {
#ifdef __SSE__
    _mm_prefetch(p, _MM_HINT_T0);
#endif
  }
};

template <bool enable_profile = false>
using UnityOp8 = UnityOp<faiss::PQDecoder8, enable_profile>;

}  // namespace detail
}  // namespace unity