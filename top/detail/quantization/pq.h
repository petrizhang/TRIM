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

#include <cassert>
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

  void set_derived_values();
  /// return the centroids associated with subvector m
  float* get_centroids(size_t m, size_t i);
  const float* get_centroids(size_t m, size_t i) const;

  /*************************************************
   * Decode/encode functions
   *************************************************/
  /// decode a vector from a given code (or n vectors if third argument)
  void decode(const uint8_t* code, float* x) const;
  void decode(const uint8_t* code, float* x, size_t n) const;
};

void ProductQuantizer::set_derived_values() {
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

float* ProductQuantizer::get_centroids(size_t m, size_t i) {
  return &centroids[(m * ksub + i) * dsub];
}

const float* ProductQuantizer::get_centroids(size_t m, size_t i) const {
  return &centroids[(m * ksub + i) * dsub];
}
/*************************************************
 * Objects to encode / decode strings of bits
 *************************************************/

struct PQEncoderGeneric {
  uint8_t* code;  ///< code for this vector
  uint8_t offset;
  const int nbits;  ///< number of bits per subquantizer index

  uint8_t reg;

  PQEncoderGeneric(uint8_t* code, int nbits, uint8_t offset = 0);

  void encode(uint64_t x);

  ~PQEncoderGeneric();
};

struct PQEncoder8 {
  uint8_t* code;
  PQEncoder8(uint8_t* code, int nbits);
  void encode(uint64_t x);
};

struct PQEncoder16 {
  uint16_t* code;
  PQEncoder16(uint8_t* code, int nbits);
  void encode(uint64_t x);
};

struct PQDecoderGeneric {
  const uint8_t* code;
  uint8_t offset;
  const int nbits;
  const uint64_t mask;
  uint8_t reg;
  PQDecoderGeneric(const uint8_t* code, int nbits);
  uint64_t decode();
};

struct PQDecoder8 {
  static const int nbits = 8;
  const uint8_t* code;
  PQDecoder8(const uint8_t* code, int nbits);
  uint64_t decode();
};

struct PQDecoder16 {
  static const int nbits = 16;
  const uint16_t* code;
  PQDecoder16(const uint8_t* code, int nbits);
  uint64_t decode();
};

template <class PQDecoder>
void decode(const ProductQuantizer& pq, const uint8_t* code, float* x) {
  PQDecoder decoder(code, pq.nbits);
  for (size_t m = 0; m < pq.M; m++) {
    uint64_t c = decoder.decode();
    memcpy(x + m * pq.dsub, pq.get_centroids(m, c), sizeof(float) * pq.dsub);
  }
}

void ProductQuantizer::decode(const uint8_t* code, float* x) const {
  switch (nbits) {
    case 8:
      top::detail::decode<PQDecoder8>(*this, code, x);
      break;

    case 16:
      top::detail::decode<PQDecoder16>(*this, code, x);
      break;

    default:
      top::detail::decode<PQDecoderGeneric>(*this, code, x);
      break;
  }
}

void ProductQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
  // #pragma omp parallel for if (n > 100)
  for (int64_t i = 0; i < n; i++) {
    this->decode(code + code_size * i, x + d * i);
  }
}

inline PQEncoderGeneric::PQEncoderGeneric(uint8_t* code, int nbits, uint8_t offset)
    : code(code), offset(offset), nbits(nbits), reg(0) {
  assert(nbits <= 64);
  if (offset > 0) {
    reg = (*code & ((1 << offset) - 1));
  }
}

inline void PQEncoderGeneric::encode(uint64_t x) {
  reg |= (uint8_t)(x << offset);
  x >>= (8 - offset);
  if (offset + nbits >= 8) {
    *code++ = reg;

    for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i) {
      *code++ = (uint8_t)x;
      x >>= 8;
    }

    offset += nbits;
    offset &= 7;
    reg = (uint8_t)x;
  } else {
    offset += nbits;
  }
}

inline PQEncoderGeneric::~PQEncoderGeneric() {
  if (offset > 0) {
    *code = reg;
  }
}

inline PQEncoder8::PQEncoder8(uint8_t* code, int nbits) : code(code) { assert(8 == nbits); }

inline void PQEncoder8::encode(uint64_t x) { *code++ = (uint8_t)x; }

inline PQEncoder16::PQEncoder16(uint8_t* code, int nbits) : code((uint16_t*)code) {
  assert(16 == nbits);
}

inline void PQEncoder16::encode(uint64_t x) { *code++ = (uint16_t)x; }

inline PQDecoderGeneric::PQDecoderGeneric(const uint8_t* code, int nbits)
    : code(code), offset(0), nbits(nbits), mask((1ull << nbits) - 1), reg(0) {
  assert(nbits <= 64);
}

inline uint64_t PQDecoderGeneric::decode() {
  if (offset == 0) {
    reg = *code;
  }
  uint64_t c = (reg >> offset);

  if (offset + nbits >= 8) {
    uint64_t e = 8 - offset;
    ++code;
    for (int i = 0; i < (nbits - (8 - offset)) / 8; ++i) {
      c |= ((uint64_t)(*code++) << e);
      e += 8;
    }

    offset += nbits;
    offset &= 7;
    if (offset > 0) {
      reg = *code;
      c |= ((uint64_t)reg << e);
    }
  } else {
    offset += nbits;
  }

  return c & mask;
}

inline PQDecoder8::PQDecoder8(const uint8_t* code, int nbits_in) : code(code) {
  assert(8 == nbits_in);
}

inline uint64_t PQDecoder8::decode() { return (uint64_t)(*code++); }

inline PQDecoder16::PQDecoder16(const uint8_t* code, int nbits_in) : code((uint16_t*)code) {
  assert(16 == nbits_in);
}

inline uint64_t PQDecoder16::decode() { return (uint64_t)(*code++); }

}  // namespace detail
}  // namespace top