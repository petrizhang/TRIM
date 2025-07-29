#pragma once

#include <algorithm>
#include <memory>
#include <queue>
#include <type_traits>
// #include <H5Cpp.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/code_distance/code_distance.h>
#include <faiss/utils/distances.h>

#include <iomanip>

#include "trim/common/common.h"
#include "trim/common/prefetch.h"
#include "trim/common/searcher.h"
#include "trim/common/setter_proxy.h"
#include "trim/detail/index/tIVFPQfs.h"

// using namespace H5;
namespace trim {
namespace detail {

struct TrimRefineResultHandler {};

struct IVFPQFastScanSearcher : SetterProxy<IVFPQFastScanSearcher>, ISearcher {
  using dist_t = float;
  using idx_t = faiss::idx_t;
  using This = IVFPQFastScanSearcher;
  using Proxy = SetterProxy<This>;
  using RangeSearchResult = faiss::RangeSearchResult;

  // Parameters
  size_t _nprobe{30};
  float _k_factor{10};
  float _gamma{0.8};
  mutable float _pruning_ratio{0.0f};
  mutable float _actual_distance_computation{0.0f};
  mutable float _total_distance_computation{0.0f};
  bool _trim_opened{true};

  // Index
  std::unique_ptr<faiss::tIVFPQfs> _ivfpqfs{nullptr};
  // faiss::tIVFPQfs* _ivfpqfs{nullptr};

  // data
  const float* _data{nullptr};

  U_FORBID_COPY_AND_ASSIGN(IVFPQFastScanSearcher);

  U_FORBID_MOVE(IVFPQFastScanSearcher);

  IVFPQFastScanSearcher() = delete;

  explicit IVFPQFastScanSearcher(const char* index_path) : Proxy("IVFPQFastScanSearcher") {
    _ivfpqfs = std::make_unique<faiss::tIVFPQfs>(index_path);
    // _ivfpqfs = new faiss::tIVFPQfs(index_path);
    _pruning_ratio = 0.0f;
    _actual_distance_computation = 0.0f;
    _total_distance_computation = 0.0f;

    Proxy::template bind<INTEGER_TYPE>("nprobe", &This::set_nprobe);
    Proxy::template bind<DOUBLE_TYPE>("gamma", &This::set_gamma);

  }

  ~IVFPQFastScanSearcher() override = default;

  void set_data(float* data) override {
    this->_data = data;
    _ivfpqfs->set_data(data);
    _ivfpqfs->compute_recons_errors();
  }

  void set(const std::string& key, const Object& value) override { Proxy::proxy_set(key, value); }

  void try_set(const std::string& key, const Object& value) override {
    Proxy::proxy_try_set(key, value);
  }

  const float* get_data(unsigned i) const override { return _ivfpqfs->get_data(i); }

  size_t num_data_points() const override { return _ivfpqfs == nullptr ? 0 : _ivfpqfs->ntotal; }

  void set_k_factor(float kf) { this->_k_factor = kf; }

  void set_nprobe(size_t nprobe) { this->_nprobe = nprobe; _ivfpqfs->nprobe = nprobe;}

  void set_gamma(float gamma) { this->_gamma = gamma; _ivfpqfs->gamma = gamma;}

  void set_trim_opened(bool flag) { this->_trim_opened = flag; }

  void optimize(int num_threads) override { T_THROW_MSG("not implemented error"); }

  IDCO* get_dco() const override { T_THROW_MSG("not implemented error"); }

  Dict get_profile() const override { T_THROW_MSG("not implemented error"); }

  float get_pruning_ratio() const override { return _pruning_ratio; }

  float get_actual_distance_computation() const override { return _actual_distance_computation; }

  float get_total_distance_computation() const override { return _total_distance_computation; }

  void clear_pruning_ratio() const override { _pruning_ratio = 0.0; }

  void clear_num_distance_computation() const override {
    _actual_distance_computation = 0.0;
    _total_distance_computation = 0.0;
  }

  void ann_search(const float* q, int k, int* dst) const override {
    // omp_set_num_threads(1);
    // std::cout << "gamma:" << _ivfpqfs->gamma << std::endl;
    // std::cout << "nprobe:" << _ivfpqfs->nprobe << std::endl;
    
    std::vector<int64_t> ids(k);
    std::vector<float> distances(k);
    // std::cout << "start query" << std::endl;
    _ivfpqfs->search(1, q, k, distances.data(), ids.data());
    // std::cout << "end query" << std::endl;

    for (int i = 0; i < k; ++i) {
      dst[i] = static_cast<int>(ids[i]);
    }
    _actual_distance_computation += _ivfpqfs->_actual_distance_computation;
    _total_distance_computation += _ivfpqfs->_total_distance_computation;
    _pruning_ratio += _ivfpqfs->_pruning_ratio;
    _ivfpqfs->clear_profile();
  }

  void range_search(const float* q, float radius, std::vector<int> &result) const override { T_THROW_MSG("not implemented error"); }
  

};
}  // namespace detail
}  // namespace trim