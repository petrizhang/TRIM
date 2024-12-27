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

#include <cstdint>
#include <vector>

#include "top/common/top_assert.h"
#include "top/detail/core/memory.h"
#include "top/detail/faiss/Index.h"

namespace top {
namespace detail {
using namespace faiss;

struct ProductQuantizer {
  size_t d;          ///< size of the input vectors
  size_t code_size;  ///< bytes per indexed vector

  size_t M;      ///< number of sub-quantizers
  size_t nbits;  ///< number of bits per quantization index

  // values derived from the above
  size_t dsub;  ///< dimensionality of each sub-vector
  size_t ksub;  ///< number of centroids for each sub-quantizer

  /// Centroid table, size M * ksub * dsub.
  /// Layout: (M, ksub, dsub)
  std::vector<float, align_alloc<float>> centroids;

  void set_derived_values() {
    // quite a few derived values
    TOP_THROW_IF_NOT_MSG(
        d % M == 0,
        "The dimension of the vector (d) should be a multiple of the number of subquantizers (M)");
    dsub = d / M;
    code_size = (nbits * M + 7) / 8;
    TOP_THROW_IF_MSG(nbits > 24, "nbits larger than 24 is not practical.");
    ksub = 1 << nbits;
    centroids.resize(d * ksub);
  }
};

struct IndexPQ : Index {
  ProductQuantizer pq;
  size_t code_size;
  /// encoded dataset, size ntotal * code_size
  std::vector<uint8_t> codes;
  /// Distances between each base vector and its centroid
  std::vector<float, align_alloc<float>> centroid_distances;

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
};

}  // namespace detail
}  // namespace top