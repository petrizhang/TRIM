#pragma once

#include <xmmintrin.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include "faiss/IndexIVFPQFastScan.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "faiss/impl/pq4_fast_scan.h"
#include "faiss/impl/simd_result_handlers.h"
#include "faiss/invlists/BlockInvertedLists.h"
#include "hnswlib/hnswlib.h"
#include "trim/common/common.h"
#include "trim/common/dco.h"
#include "trim/detail/io/read_faiss.h"

namespace faiss {

struct TrimRefineResultHandler
    : simd_result_handlers::ResultHandlerCompare<CMax<float, int64_t>, true> {
  using ParentClass = simd_result_handlers::ResultHandlerCompare<CMax<float, int64_t>, true>;

  //===--------------------------------------------------------------------===//
  // Search parameters and results
  //===--------------------------------------------------------------------===//
  int64_t _k;
  float* _result_dis;
  int64_t* _result_ids;
  const float* _recons_errors;  // reconstruction errors for this list
  float _gamma = 0.8;
  std::priority_queue<std::pair<float, faiss::idx_t>, std::vector<std::pair<float, faiss::idx_t>>>
      _results_queue;

  size_t _d;
  const float* _data;
  const float* _query;

  //===--------------------------------------------------------------------===//
  // Intermediate data buffers
  //===--------------------------------------------------------------------===//
  static constexpr size_t bbs = 32;
  float PORTABLE_ALIGN32 _lowerbounds[bbs];

  //===--------------------------------------------------------------------===//
  // Profile
  //===--------------------------------------------------------------------===//
  mutable float _pruning_ratio{0.0f};
  mutable float _actual_distance_computation{0.0f};
  mutable float _total_distance_computation{0.0f};
  mutable int falsePositive{0};
  //===--------------------------------------------------------------------===//
  // Constructors and methods
  //===--------------------------------------------------------------------===//
  TrimRefineResultHandler(size_t ntotal, int64_t k, float* dis, int64_t* ids, size_t d,
                          const float* data, const float* query)
      : ParentClass(1, ntotal, nullptr),
        _k(k),
        _result_dis(dis),
        _result_ids(ids),
        _d(d),
        _data(data),
        _query(query) {}

  void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
    FAISS_ASSERT(_recons_errors != nullptr);
    // auto start = std::chrono::high_resolution_clock::now();
    this->adjust_with_origin(q, d0, d1);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "adjust_with_origin: "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //           << " ms" << std::endl;
    // load_pq_dist(d0, d1);
    // compute_lowerbounds(b);
    // start = std::chrono::high_resolution_clock::now();
    compute_lowerbounds_simd(d0, d1, b);
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "compute_lowerbounds_simd: "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //           << " ms" << std::endl;
    // start = std::chrono::high_resolution_clock::now();
    refine(b);
    // end = std::chrono::high_resolution_clock::now();
    // std::cout << "refine: "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //           << " ms" << std::endl;
  }

  void end() override {
    size_t i = _results_queue.size() - 1;
    while (!_results_queue.empty()) {
      auto [dist, id] = _results_queue.top();
      _results_queue.pop();
      _result_dis[i] = dist;
      _result_ids[i] = id;
      --i;
    }
    _pruning_ratio = 1 - _actual_distance_computation / _total_distance_computation;
  }

  // compute and adjust idx
  int64_t adjust_id(size_t b, size_t j) {
    int64_t idx = j0 + 32 * b + j;
    if (idx >= ntotal) {
      return -1;
    }
    idx = id_map[idx];
    return idx;
  }

  void refine(size_t b) {
    _total_distance_computation += bbs;
    constexpr size_t mini_batch_size = 8;
    for (size_t j = 0; j < bbs / mini_batch_size; j++) {
      trim::Bool8 bools(true);
      __m256 lb_batch = _mm256_load_ps(_lowerbounds + j * mini_batch_size);
      if (LIKELY(!_results_queue.empty())) {
        __m256 threshold = _mm256_set1_ps(_results_queue.top().first);
        __m256 cmp_result = _mm256_cmp_ps(lb_batch, threshold, _CMP_LT_OQ);
        int mask = _mm256_movemask_ps(cmp_result);
        bools = trim::Bool8(mask);
        if (LIKELY(!bools.has_true())) {
          continue;
        }
      }

      for (size_t i = 0; i < 8; i++) {
        if (bools.get(i)) {
          float max_dist = _results_queue.size() < _k ? std::numeric_limits<float>::max()
                                                      : _results_queue.top().first;
          auto lowerbound = lb_batch[i];
          if (lowerbound < max_dist) {
            auto id = adjust_id(b, i + j * mini_batch_size);
            if (id < 0) {
              continue;
            }
            _actual_distance_computation++;
            float dist = fvec_L2sqr(_query, id);
            if (dist < max_dist) {
              _results_queue.emplace(dist, id);
              if (_results_queue.size() > _k) _results_queue.pop();
            }
          }
        }
      }
    }
  }

  static __m256 dequantize(__m256 dis, __m256 vec_a, __m256 vec_b) {
#if defined(__FMA__) || defined(__AVF512F__)
    return _mm256_fmadd_ps(dis, vec_a, vec_b);
#else
    return dis * vec_a + vec_b;
#endif
  }

  void compute_lowerbounds_simd(simd16uint16 d0, simd16uint16 d1, size_t batch_round) {
    float a = 1.0, b = 0.0;
    if (normalizers) {
      a = 1 / normalizers[0];
      b = normalizers[1];
    }

    const float* __restrict landmark_dists = _recons_errors + j0 + batch_round * bbs;
    __m256 vec_a = _mm256_set1_ps(a);
    __m256 vec_b = _mm256_set1_ps(b);
    const float gamma = _gamma;
    // Process d0
    __m256 float_vec_lo0, float_vec_hi0;
    convert_16x_uint16_to_2x8_float32(d0.i, &float_vec_lo0, &float_vec_hi0);
    float_vec_lo0 = dequantize(float_vec_lo0, vec_a, vec_b);
    __m256 float_vec_lo_sqrt0 = _mm256_sqrt_ps(float_vec_lo0);
    __m256 recons_error_vec_lo0 = _mm256_load_ps(landmark_dists);
    __m256 lb_vec_lo0 = _relaxed_lowerbound8_avx2(gamma, float_vec_lo_sqrt0, recons_error_vec_lo0);
    _mm256_store_ps(_lowerbounds, lb_vec_lo0);

    float_vec_hi0 = dequantize(float_vec_hi0, vec_a, vec_b);
    __m256 float_vec_hi_sqrt0 = _mm256_sqrt_ps(float_vec_hi0);
    __m256 recons_error_vec_hi0 = _mm256_load_ps(landmark_dists + 8);
    __m256 lb_vec_hi0 = _relaxed_lowerbound8_avx2(gamma, float_vec_hi_sqrt0, recons_error_vec_hi0);
    _mm256_store_ps(_lowerbounds + 8, lb_vec_hi0);

    // Process d1
    __m256 float_vec_lo1, float_vec_hi1;
    convert_16x_uint16_to_2x8_float32(d1.i, &float_vec_lo1, &float_vec_hi1);
    float_vec_lo1 = dequantize(float_vec_lo1, vec_a, vec_b);
    __m256 float_vec_lo_sqrt1 = _mm256_sqrt_ps(float_vec_lo1);
    __m256 recons_error_vec_lo1 = _mm256_load_ps(landmark_dists + 16);
    __m256 lb_vec_lo1 = _relaxed_lowerbound8_avx2(gamma, float_vec_lo_sqrt1, recons_error_vec_lo1);
    _mm256_store_ps(_lowerbounds + 16, lb_vec_lo1);

    float_vec_hi1 = dequantize(float_vec_hi1, vec_a, vec_b);
    __m256 float_vec_hi_sqrt1 = _mm256_sqrt_ps(float_vec_hi1);
    __m256 recons_error_vec_hi1 = _mm256_load_ps(landmark_dists + 24);
    __m256 lb_vec_hi1 = _relaxed_lowerbound8_avx2(gamma, float_vec_hi_sqrt1, recons_error_vec_hi1);
    _mm256_store_ps(_lowerbounds + 24, lb_vec_hi1);
  }

  void convert_16x_uint16_to_2x8_float32(__m256i uint16_vec, __m256* float_vec_lo,
                                         __m256* float_vec_hi) {
    // 1. 提取低8个uint16和高8个uint16
    __m128i uint16_lo = _mm256_extracti128_si256(uint16_vec, 0);  // 低128位
    __m128i uint16_hi = _mm256_extracti128_si256(uint16_vec, 1);  // 高128位

    // 2. 将低8个uint16扩展为8个uint32并转换为float32
    __m256i uint32_lo = _mm256_cvtepu16_epi32(uint16_lo);
    *float_vec_lo = _mm256_cvtepi32_ps(uint32_lo);

    // 3. 将高8个uint16扩展为8个uint32并转换为float32
    __m256i uint32_hi = _mm256_cvtepu16_epi32(uint16_hi);
    *float_vec_hi = _mm256_cvtepi32_ps(uint32_hi);
  }

  const float* get_data(size_t i) const { return _data + i * _d; }

  float fvec_L2sqr(const float* x, idx_t id) const {
    size_t d = this->_d;
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

    float PORTABLE_ALIGN32 res[8];
    _mm256_store_ps(res, sum);
    float result = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];

    for (; i < d; i++) {
      const float tmp = x[i] - y[i];
      result += tmp * tmp;
    }

    return result;
  }

  __always_inline __m256 _relaxed_lowerbound8_avx2(float gamma, __m256 pq_dist_vec,
                                                   __m256 recons_error_vec) {
    // Squared root of PQ distances
    __m256 vec_a = pq_dist_vec;
    // Reconstruction errors (actully squared roots of reconstruction errors)
    __m256 vec_b = recons_error_vec;
    // lowerbounds = (a[i] - b[i])^2 + 2 * gamma * a[i] * b[i]
    __m256 vec_diff = _mm256_sub_ps(vec_a, vec_b);
    __m256 vec_diff_squared = _mm256_mul_ps(vec_diff, vec_diff);
    __m256 vec_2gamma = _mm256_set1_ps(2 * gamma);
    __m256 vec_ab = _mm256_mul_ps(vec_a, vec_b);
    __m256 vec_2gamma_ab = _mm256_mul_ps(vec_2gamma, vec_ab);
    __m256 lowerbounds = _mm256_add_ps(vec_diff_squared, vec_2gamma_ab);
    return lowerbounds;
  }
};

struct tIVFPQfs : IndexIVFPQFastScan {
  std::vector<AlignedTable<float, 64>> perlist_recons_errors;
  float* data{nullptr};
  float gamma = 0.8;
  //===--------------------------------------------------------------------===//
  // Profile
  //===--------------------------------------------------------------------===//
  mutable float _pruning_ratio{0.0f};
  mutable float _actual_distance_computation{0.0f};
  mutable float _total_distance_computation{0.0f};

  //===--------------------------------------------------------------------===//
  // Constructors and methods
  //===--------------------------------------------------------------------===//
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
      ivpq->fine_quantizer = &ivpq->pq;
      // ivpq->init_fastscan(M, nbits, nlist, metric_type, bbs);
    } else
      throw std::invalid_argument("Wrong index type");

    if (this->by_residual) {
      throw std::invalid_argument("by_residual should be false");
    }

    this->metric_type = METRIC_L2;
    // this->parallel_mode = PARALLEL_MODE_NO_HEAP_INIT;
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

  //===--------------------------------------------------------------------===//
  // Search algorithms
  //===--------------------------------------------------------------------===//
  static size_t roundup(size_t a, size_t b) { return (a + b - 1) / b * b; }

  using CoarseQuantized = IndexIVFFastScan::CoarseQuantized;

  struct CoarseQuantizedWithBuffer : CoarseQuantized {
    explicit CoarseQuantizedWithBuffer(const CoarseQuantized& cq) : CoarseQuantized(cq) {}

    bool done() const { return ids != nullptr; }

    std::vector<idx_t> ids_buffer;
    std::vector<float> dis_buffer;

    void quantize(const Index* quantizer, idx_t n, const float* x,
                  const SearchParameters* quantizer_params) {
      dis_buffer.resize(nprobe * n);
      ids_buffer.resize(nprobe * n);
      quantizer->search(n, x, nprobe, dis_buffer.data(), ids_buffer.data(), quantizer_params);
      dis = dis_buffer.data();
      ids = ids_buffer.data();
    }
  };

  struct CoarseQuantizedSlice : CoarseQuantizedWithBuffer {
    size_t i0, i1;
    CoarseQuantizedSlice(const CoarseQuantized& cq, size_t i0, size_t i1)
        : CoarseQuantizedWithBuffer(cq), i0(i0), i1(i1) {
      if (done()) {
        dis += nprobe * i0;
        ids += nprobe * i0;
      }
    }

    void quantize_slice(const Index* quantizer, const float* x,
                        const SearchParameters* quantizer_params) {
      quantize(quantizer, i1 - i0, x + quantizer->d * i0, quantizer_params);
    }
  };

  int compute_search_nslice(const IndexIVFFastScan* index, size_t n, size_t nprobe) {
    int nslice;
    if (n <= omp_get_max_threads()) {
      nslice = n;
    } else if (index->lookup_table_is_3d()) {
      // make sure we don't make too big LUT tables
      size_t lut_size_per_query =
          index->M * index->ksub * nprobe * (sizeof(float) + sizeof(uint8_t));

      size_t max_lut_size = precomputed_table_max_bytes;
      // how many queries we can handle within mem budget
      size_t nq_ok = std::max(max_lut_size / lut_size_per_query, size_t(1));
      nslice = roundup(std::max(size_t(n / nq_ok), size_t(1)), omp_get_max_threads());
    } else {
      // LUTs unlikely to be a limiting factor
      nslice = omp_get_max_threads();
    }
    return nslice;
  }

  void search_preassigned(idx_t n, const float* x, idx_t k, const idx_t* assign,
                          const float* centroid_dis, float* distances, idx_t* labels,
                          bool store_pairs, const IVFSearchParameters* params,
                          IndexIVFStats* stats) const override {
    size_t nprobe = this->nprobe;
    if (params) {
      FAISS_THROW_IF_NOT(params->max_codes == 0);
      nprobe = params->nprobe;
    }

    FAISS_THROW_IF_NOT_MSG(!store_pairs, "store_pairs not supported for this index");
    FAISS_THROW_IF_NOT_MSG(!stats, "stats not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);

    const CoarseQuantized cq = {nprobe, centroid_dis, assign};
    search_dispatch_implem(n, x, k, distances, labels, cq, nullptr, params);
  }

  void search_dispatch_implem(idx_t n, const float* x, idx_t k, float* distances, idx_t* labels,
                              const CoarseQuantized& cq_in, const NormTableScaler* scaler,
                              const IVFSearchParameters* params) const {
    FAISS_ASSERT(bbs == 32);

    const idx_t nprobe = params ? params->nprobe : this->nprobe;
    const IDSelector* sel = (params) ? params->sel : nullptr;
    const SearchParameters* quantizer_params = params ? params->quantizer_params : nullptr;

    bool is_max = !is_similarity_metric(metric_type);
    FAISS_ASSERT(is_max);
    using RH = SIMDResultHandlerToFloat;

    if (n == 0) {
      return;
    }

    // actual implementation used
    int impl = 12;

    CoarseQuantizedWithBuffer cq(cq_in);
    cq.nprobe = nprobe;

    if (!cq.done()) {
      // we do the coarse quantization here execpt when search is
      // sliced over threads (then it is more efficient to have each thread do
      // its own coarse quantization)
      cq.quantize(quantizer, n, x, quantizer_params);
      invlists->prefetch_lists(cq.ids, n * cq.nprobe);
    }

    size_t ndis = 0, nlist_visited = 0;
    auto handler =
        std::make_unique<TrimRefineResultHandler>(ntotal, k, distances, labels, d, data, x);
    handler->_gamma = gamma;
    search_implem_10(n, x, *handler.get(), cq, &ndis, &nlist_visited, scaler, params);
    _actual_distance_computation = handler->_actual_distance_computation;
    _total_distance_computation = handler->_total_distance_computation;
    _pruning_ratio = handler->_pruning_ratio;
  }

  void search_implem_10(idx_t n, const float* x, SIMDResultHandlerToFloat& handler,
                        const CoarseQuantized& cq, size_t* ndis_out, size_t* nlist_out,
                        const NormTableScaler* scaler, const IVFSearchParameters* params) const {
    size_t dim12 = ksub * M2;
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::unique_ptr<float[]> normalizers(new float[2 * n]);

    compute_LUT_uint8(n, x, cq, dis_tables, biases, normalizers.get());

    bool single_LUT = !lookup_table_is_3d();

    size_t ndis = 0;
    int qmap1[1];

    handler.q_map = qmap1;
    handler.begin(skip & 16 ? nullptr : normalizers.get());
    size_t nprobe = cq.nprobe;

    for (idx_t i = 0; i < n; i++) {
      const uint8_t* LUT = nullptr;
      qmap1[0] = i;

      if (single_LUT) {
        LUT = dis_tables.get() + i * dim12;
      }
      for (idx_t j = 0; j < nprobe; j++) {
        size_t ij = i * nprobe + j;
        if (!single_LUT) {
          LUT = dis_tables.get() + ij * dim12;
        }
        if (biases.get()) {
          handler.dbias = biases.get() + ij;
        }

        idx_t list_no = cq.ids[ij];
        if (list_no < 0) {
          continue;
        }
        size_t ls = invlists->list_size(list_no);
        if (ls == 0) {
          continue;
        }

        InvertedLists::ScopedCodes codes(invlists, list_no);
        InvertedLists::ScopedIds ids(invlists, list_no);

        handler.ntotal = ls;
        handler.id_map = ids.get();

        ((TrimRefineResultHandler&)handler)._recons_errors = perlist_recons_errors[list_no].data();
        pq4_accumulate_loop(1, roundup(ls, bbs), bbs, M2, codes.get(), LUT, handler, scaler);

        ndis++;
      }
    }

    handler.end();
    *ndis_out = ndis;
    *nlist_out = nlist;
  }

  // TODO: fix bug of this method
  void compute_recons_errors() {
    FAISS_ASSERT(data != nullptr);

    perlist_recons_errors.resize(nlist);
    for (size_t i = 0; i < nlist; i++) {
      size_t list_size = this->get_list_size(i);
      size_t target_size = roundup(list_size, bbs);
      auto& invlist_errors = perlist_recons_errors[i];
      invlist_errors.resize(target_size);
      memset(invlist_errors.data(), 0, target_size * sizeof(float));
    }

    if (!invlists) {
      std::cerr << "Error: invlists is null." << std::endl;
      return;
    }

    std::vector<float> x_approx(d);
    for (int list_no = 0; list_no < nlist; list_no++) {
      size_t list_size = invlists->list_size(list_no);

      if (list_size == 0) {
        continue;
      }

      auto& invlist_errors = perlist_recons_errors[list_no];
      const idx_t* ids = invlists->get_ids(list_no);
      for (size_t offset = 0; offset < list_size; offset++) {
        idx_t id = ids[offset];
        reconstruct_from_offset(list_no, offset, x_approx.data());
        float error = fvec_L2sqr(x_approx.data(), id);
        invlist_errors[offset] = std::sqrt(error);
      }
    }
  }

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

    float PORTABLE_ALIGN32 res[8];
    _mm256_store_ps(res, sum);
    float result = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];

    for (; i < d; i++) {
      const float tmp = x[i] - y[i];
      result += tmp * tmp;
    }

    return result;
  }

  void clear_profile() {
    _pruning_ratio = 0.0;
    _actual_distance_computation = 0.0;
    _total_distance_computation = 0.0;
  }
};

}  // namespace faiss