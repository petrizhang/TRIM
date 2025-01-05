/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <top/common/top_assert.h>
#include <top/detail/faiss/Index.h>
#include <top/detail/faiss/utils/distances_simd.h>

#include <cassert>
#include <cstdint>
#include <vector>

namespace faiss {

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
  std::vector<float> centroids;

  void set_derived_values();
  float* get_centroids(size_t m, size_t i);
  const float* get_centroids(size_t m, size_t i) const;
  
  void compute_distance_table(const float* x, float* dis_table) const;
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

inline void ProductQuantizer::compute_distance_table(const float* x, float* dis_table) const {
  // use regular version
  for (size_t m = 0; m < M; m++) {
    fvec_L2sqr_ny(dis_table + m * ksub, x + m * dsub, get_centroids(m, 0), dsub, ksub);
  }
}

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

inline void ProductQuantizer::decode(const uint8_t* code, float* x) const {
  switch (nbits) {
    case 8:
      faiss::decode<PQDecoder8>(*this, code, x);
      break;

    case 16:
      faiss::decode<PQDecoder16>(*this, code, x);
      break;

    default:
      faiss::decode<PQDecoderGeneric>(*this, code, x);
      break;
  }
}

inline void ProductQuantizer::decode(const uint8_t* code, float* x, size_t n) const {
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

}  // namespace faiss