#pragma once

#include <xmmintrin.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "faiss/IndexIVFPQFastScan.h"
#include "faiss/impl/io.h"
#include "faiss/impl/io_macros.h"
#include "faiss/impl/simd_result_handlers.h"
#include "faiss/invlists/BlockInvertedLists.h"
#include "trim/detail/io/read_faiss.h"

namespace faiss {

struct floatx32 {
  std::array<float, 32> values;
};

struct int64x32 {
  std::array<int64_t, 32> values;
};

struct TrimRefineResultHandler
    : simd_result_handlers::ResultHandlerCompare<CMax<float, int64_t>, true> {
  using ParentClass = simd_result_handlers::ResultHandlerCompare<CMax<float, int64_t>, true>;

  //===--------------------------------------------------------------------===//
  // Search parameters and results
  //===--------------------------------------------------------------------===//
  int64_t _k;
  float* _result_dis;
  int64_t* _result_ids;
  const float* _global_recons_errors;
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
  std::array<int64_t, bbs> _ids;
  std::array<float, bbs> _pq_dist_sqrt;
  std::array<float, bbs> _recons_error_sqrt;
  std::array<float, bbs> _lowerbounds;

  //===--------------------------------------------------------------------===//
  // Profile
  //===--------------------------------------------------------------------===//
  mutable float _pruning_ratio{0.0f};
  mutable float _actual_distance_computation{0.0f};
  mutable float _total_distance_computation{0.0f};

  //===--------------------------------------------------------------------===//
  // Constructors and methods
  //===--------------------------------------------------------------------===//
  TrimRefineResultHandler(size_t ntotal, int64_t k, float* dis, int64_t* ids,
                          const float* global_recons_errors, size_t d, const float* data,
                          const float* query)
      : ParentClass(1, ntotal, nullptr),
        _k(k),
        _result_dis(dis),
        _result_ids(ids),
        _global_recons_errors(global_recons_errors),
        _d(d),
        _data(data),
        _query(query) {
    FAISS_ASSERT(global_recons_errors != nullptr);
  }

  void handle(size_t q, size_t b, simd16uint16 d0, simd16uint16 d1) final {
    this->adjust_with_origin(q, d0, d1);
    load_id(b);
    load_recons_errors();
    load_pq_dist(d0, d1);
    compute_lowerbounds();
    refine();
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
    if (idx > ntotal) {
      return -1;
    }
    idx = id_map[idx];
    return idx;
  }

  void refine() {
    _total_distance_computation += bbs;
    for (size_t i = 0; i < bbs; i++) {
      auto lowerbound = _lowerbounds[i];
      auto id = _ids[i];
      if (id < 0) {
        continue;
      }
      if (_results_queue.size() < _k || lowerbound < _results_queue.top().first) {
        _actual_distance_computation++;
        float dist = fvec_L2sqr(_query, id);
        _results_queue.emplace(dist, id);
        if (_results_queue.size() > _k) _results_queue.pop();
      }
    }
  }

  void load_recons_errors() {
    const float* __restrict recons_errors = _global_recons_errors;
    float* __restrict dest = _recons_error_sqrt.data();
    for (size_t i = 0; i < bbs; i++) {
      int64_t id = _ids[i];
      float error = recons_errors[id];
      dest[i] = error;
    }
  }

  void compute_lowerbounds() {
    const float* __restrict data_a = _pq_dist_sqrt.data();
    const float* __restrict data_b = _recons_error_sqrt.data();
    const float gamma_value = _gamma;
    for (size_t i = 0; i < bbs; i++) {
      float ai = data_a[i];
      float bi = data_b[i];
      _lowerbounds[i] = (ai - bi) * (ai - bi) + 2 * gamma_value * ai * bi;
    }
  }

  void load_id(size_t b) {
    for (int64_t j = 0; j < bbs; ++j) {
      int64_t id = adjust_id(b, j);
      _ids[j] = id;
    }
  }

  void load_pq_dist(simd16uint16 d0, simd16uint16 d1) {
    float a = 1.0, b = 0.0;
    if (normalizers) {
      a = 1 / normalizers[0];
      b = normalizers[1];
    }
    __m256 vec_a = _mm256_set1_ps(a);
    __m256 vec_b = _mm256_set1_ps(b);

    auto dequantize = [vec_a, vec_b](__m256 dis) { return dis * vec_a + vec_b; };

    // Process d0
    __m256 float_vec_lo0, float_vec_hi0;
    convert_16x_uint16_to_2x8_float32(d0.i, &float_vec_lo0, &float_vec_hi0);
    float_vec_lo0 = dequantize(float_vec_lo0);
    __m256 float_vec_lo_sqrt0 = _mm256_sqrt_ps(float_vec_lo0);
    _mm256_storeu_ps(_pq_dist_sqrt.data(), float_vec_lo_sqrt0);
    float_vec_hi0 = dequantize(float_vec_hi0);
    __m256 float_vec_hi_sqrt0 = _mm256_sqrt_ps(float_vec_hi0);
    _mm256_storeu_ps(_pq_dist_sqrt.data() + 8, float_vec_hi_sqrt0);

    // Process d1
    __m256 float_vec_lo1, float_vec_hi1;
    convert_16x_uint16_to_2x8_float32(d1.i, &float_vec_lo1, &float_vec_hi1);
    float_vec_lo1 = dequantize(float_vec_lo1);
    __m256 float_vec_lo_sqrt1 = _mm256_sqrt_ps(float_vec_lo1);
    _mm256_storeu_ps(_pq_dist_sqrt.data() + 16, float_vec_lo_sqrt1);
    float_vec_hi1 = dequantize(float_vec_hi1);
    __m256 float_vec_hi_sqrt1 = _mm256_sqrt_ps(float_vec_hi1);
    _mm256_storeu_ps(_pq_dist_sqrt.data() + 24, float_vec_hi_sqrt1);
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

    float res[8];
    _mm256_storeu_ps(res, sum);
    float result = res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];

    for (; i < d; i++) {
      const float tmp = x[i] - y[i];
      result += tmp * tmp;
    }

    return result;
  }
};

struct tIVFPQfs : IndexIVFPQFastScan {
  std::vector<float> recons_errors;
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
    auto handler = std::make_unique<TrimRefineResultHandler>(ntotal, k, distances, labels,
                                                             recons_errors.data(), d, data, x);
    handler->_gamma = gamma;
    search_implem_10(n, x, *handler.get(), cq, &ndis, &nlist_visited, scaler, params);
    _actual_distance_computation = handler->_actual_distance_computation;
    _total_distance_computation = handler->_total_distance_computation;
    _pruning_ratio = handler->_pruning_ratio;
  }

  // TODO: fix bug of this method
  void compute_recons_errors() {
    FAISS_ASSERT(data != nullptr);
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
};

}  // namespace faiss