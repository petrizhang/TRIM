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

#include <cmath>
#include <future>
#include <vector>

#include "unity/detail/faiss/impl/ProductQuantizer.h"
#include "unity/detail/faiss/utils/AlignedTable.h"
#include "unity/detail/hnswlib/hnswlib.h"
#include "unity/util/thread_pool.h"

namespace unity {
namespace detail {
using namespace faiss;

struct IndexPQ : Index {
  ProductQuantizer pq;
  size_t code_size;
  /// encoded dataset, size ntotal * code_size
  AlignedTable<uint8_t> codes;
  /// Distances between each base vector and its centroid
  AlignedTable<float> recons_errors;

  /******************************************************
   * Polysemous codes implementation, currently not supported by TOP
   ******************************************************/
  bool do_polysemous_training;  ///< false = standard PQ

  /// how to perform the search in search_core
  enum Search_type_t {
    ST_PQ,                     ///< asymmetric product quantizer (default)
    ST_HE,                     ///< Hamming distance on codes
    ST_generalized_HE,         ///< nb of same codes
    ST_SDC,                    ///< symmetric product quantizer (SDC)
    ST_polysemous,             ///< HE filter (using ht) + PQ combination
    ST_polysemous_generalize,  ///< Filter on generalized Hamming
  };

  Search_type_t search_type;

  // just encode the sign of the components, instead of using the PQ encoder
  // used only for the queries
  bool encode_signs;

  /// Hamming threshold used for polysemy
  int polysemous_ht;

  /******************************************************
   * Decode/encode functions
   ******************************************************/
  void sa_decode_n(idx_t n, const uint8_t* bytes, float* x) const;
  void reconstruct_n(idx_t i0, idx_t ni, float* recons) const;
  void sa_decode(const uint8_t* code, float* x) const;
  void reconstruct(idx_t key, float* recons) const;
};

inline void IndexPQ::sa_decode_n(idx_t n, const uint8_t* bytes, float* x) const {
  pq.decode(bytes, x, n);
}

inline void IndexPQ::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
  TOP_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
  sa_decode_n(ni, codes.data() + i0 * code_size, recons);
}

inline void IndexPQ::sa_decode(const uint8_t* code, float* x) const { pq.decode(code, x); }

inline void IndexPQ::reconstruct(idx_t key, float* recons) const {
  TOP_THROW_IF_NOT(key >= 0 && key <= ntotal);
  sa_decode(codes.data() + key * code_size, recons);
}

}  // namespace detail
}  // namespace unity