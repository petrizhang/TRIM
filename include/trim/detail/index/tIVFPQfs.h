#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "faiss/IndexIVFPQFastScan.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "faiss/invlists/BlockInvertedLists.h"
#include "trim/detail/io/read_faiss.h"

namespace faiss {

struct tIVFPQfs : IndexIVFPQFastScan {
  std::vector<float> recons_errors;
  float* data{nullptr};

  // load IVFPQ
  explicit tIVFPQfs(const char* index_path) {
    // load IVFPQ
    FileIOReader reader(index_path);
    IOReader* f = &reader;
    uint32_t h;
    int io_flags = 0;
    READ1(h);
    if (h == fourcc("IwPf")) {
      IndexIVFPQFastScan* ivpq = this;
      trim_read_ivf_header(ivpq, f);
      READ1(ivpq->by_residual);
      READ1(ivpq->code_size);
      READ1(ivpq->bbs);
      READ1(ivpq->M2);
      READ1(ivpq->implem);
      READ1(ivpq->qbs2);
      trim_read_ProductQuantizer(&ivpq->pq, f);
      trim_read_InvertedLists(ivpq, f, io_flags);
      ivpq->precompute_table();

      const auto& pq = ivpq->pq;
      ivpq->M = pq.M;
      ivpq->nbits = pq.nbits;
      ivpq->ksub = (1 << pq.nbits);
      ivpq->code_size = pq.code_size;
      ivpq->init_code_packer();
      ivpq->fine_quantizer = (Quantizer*)ivpq->quantizer;
      // ivpq->init_fastscan(M, nbits, nlist, metric_type, bbs);
    } else
      throw std::invalid_argument("Wrong index type");

    if (this->by_residual) {
      throw std::invalid_argument("by_residual should be false");
    }

    this->metric_type = METRIC_L2;
    // this->parallel_mode = PARALLEL_MODE_NO_HEAP_INIT;
    recons_errors.resize(ntotal, 0.0);
  }

  ~tIVFPQfs() {}

  void set_data(float* data) { this->data = data; }

  const float* get_data(unsigned i) const {
    if (i < 0 || i >= ntotal) {
      throw std::out_of_range("Vector index out of range: " + std::to_string(i));
    }
    // float* vec = new float[ivfpq->d];
    // std::memcpy(vec, data + i * ivfpq->d, ivfpq->d * sizeof(float));

    return data + i * d;
  }

  // TODO: fix bug of this method
  void compute_recons_errors() {
    // FAISS_ASSERT(data != nullptr);
    if (!invlists) {
      std::cerr << "Error: invlists is null." << std::endl;
      return;
    }

    BlockInvertedLists* bil = dynamic_cast<BlockInvertedLists*>(invlists);
    std::vector<float> x_approx(d);
    for (int list_no = 0; list_no < nlist; list_no++) {
      size_t list_size = bil->list_size(list_no);

      if (list_size == 0) {
        continue;
      }

      const idx_t* ids = bil->get_ids(list_no);
      for (size_t i = 0; i < list_size; i++) {
        idx_t id = ids[i];
        reconstruct_from_offset(list_no, i, x_approx.data());
        float error = fvec_L2sqr(x_approx.data(), id);
        recons_errors[id] = std::sqrt(error);
      }
    }
  }

#ifdef USE_AVX
  float fvec_L2sqr(const float* x, idx_t id) const {
    size_t d = this->d;
    const float* y = get_data(id);

    __m256 sum = _mm256_setzero_ps();
    size_t i;

    for (i = 0; i < d - 7; i += 8) {
      __m256 x_vec = _mm256_loadu_ps(x + i);
      __m256 y_vec = _mm256_loadu_ps(y + i);
      __m256 diff = _mm256_sub_ps(x_vec, y_vec);
      __m256 sq = _mm256_mul_ps(diff, diff);
      sum = _mm256_add_ps(sum, sq);
    }

    float res[8];
    _mm256_storeu_ps(res, sum);
    float result = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];

    for (; i < d; i++) {
      const float tmp = x[i] - y[i];
      result += tmp * tmp;
    }

    return result;
  }
#else
  float fvec_L2sqr(const float* x, idx_t id) const {
    const float* y = get_data(id);

    size_t i;
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (i = 0; i < this->d; i++) {
      const float tmp = x[i] - y[i];
      res += tmp * tmp;
    }

    return res;
  }
#endif
};

}  // namespace faiss