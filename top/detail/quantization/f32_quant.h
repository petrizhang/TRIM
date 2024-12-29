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

#include "top/common/common.h"
#include "top/detail/core/distance.h"
#include "top/detail/core/memory.h"

namespace top {
namespace detail {

struct FP32Quantizer {
  using data_type = float;
  constexpr static int kAlign = 16;
  int d, d_align;
  int64_t code_size;
  char* codes = nullptr;

  FP32Quantizer() = default;

  explicit FP32Quantizer(int dim)
      : d(dim), d_align(do_align(dim, kAlign)), code_size(d_align * 4) {}

  ~FP32Quantizer() { free(codes); }

  void train(const float* data, int64_t n) {
    codes = (char*)alloc2M(n * code_size);
    for (int64_t i = 0; i < n; ++i) {
      encode(data + i * d, get_data(i));
    }
  }

  void encode(const float* from, char* to) { std::memcpy(to, from, d * 4); }

  char* get_data(int u) const { return codes + u * code_size; }

  template <typename Pool>
  void reorder(const Pool& pool, const float*, int* dst, int k) const {
    for (int i = 0; i < k; ++i) {
      dst[i] = pool.id(i);
    }
  }

  struct Computer {
    using dist_type = float;
    const FP32Quantizer& quant;
    float* q = nullptr;
    Computer(const FP32Quantizer& quant, const float* query)
        : quant(quant), q((float*)alloc64B(quant.d_align * 4)) {
      std::memcpy(q, query, quant.d * 4);
    }
    ~Computer() { free(q); }
    dist_type operator()(int u) const { return L2Sqr(q, (data_type*)quant.get_data(u), quant.d); }
    void prefetch(int u, int lines) const { mem_prefetch(quant.get_data(u), lines); }
  };

  auto get_computer(const float* query) const { return Computer(*this, query); }
};

}  // namespace detail
}  // namespace top
