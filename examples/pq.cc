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

#include <algorithm>
#include <iomanip>
#include <iostream>

#include "unity/detail/io/read_faiss.h"
#include "unity/detail/quantization/index_pq.h"
#include "unity/detail/uhnsw/dco_unity.h"
#include "unity/detail/uhnsw/hnsw_searcher.h"
#include "unity/unity.h"

void bench(const unity::Searcher* searcher, double gamma, size_t n_test) {
  size_t nb = searcher->num_data_points();
  auto* dco = searcher->get_dco();
  dco->set("gamma", gamma);

  n_test = std::min(nb, n_test);
  size_t n_pair = n_test * n_test;
  std::vector<float> distances(n_pair);
  std::vector<float> lowerbounds(n_pair);

  for (size_t i = 0; i < n_test; i++) {
    const float* query = searcher->get_data(i);
    dco->set_query(query);
    for (size_t j = 0; j < n_test; j++) {
      distances[i + j * n_test] = dco->compute(j);
      lowerbounds[i + j * n_test] = dco->relaxed_lowerbound(j);
    }
  }

  double mse = 0, rmae = 0;
  size_t success = 0;
  for (size_t i = 0; i < n_pair; i++) {
    if (distances[i] - lowerbounds[i] < -10e-9) {
      if (gamma == 0) {
        U_THROW_MSG("test failed");
      }
    } else {
      success += 1;
    }

    mse += double(distances[i] - lowerbounds[i]) * double(distances[i] - lowerbounds[i]);
    if (distances[i] > 10e-9) {
      rmae += std::abs(double(distances[i] - lowerbounds[i])) / double(distances[i]);
    }
  }

  std::cout << "Results for gamma=" << std::setprecision(4) << gamma
            << ": p=" << double(success) / n_pair << ", mse=" << mse / n_pair
            << ", rmae=" << rmae / n_pair << "\n";
}

int main() {
  using unity::Searcher;
  using unity::detail::HNSWSearcher;
  using unity::detail::Index;
  using unity::detail::IndexPQ;
  using unity::detail::IndexType;

  // Search hnswlib index with top
  const char* index_hnsw_path = "/data/home/petrizhang/develop/TOP/test/hnswlib.bin";
  const char* index_pq_path = "/data/home/petrizhang/develop/TOP/test/index_pq.bin";
  const int dim = 256;
  std::unique_ptr<unity::Searcher> searcher = unity::SearcherCreator(unity::constants::U_HNSW)
                                                  .set("hnswlib_index_path", index_hnsw_path)
                                                  .set("pq_index_path", index_pq_path)
                                                  .set("dim", dim)
                                                  .set("metric", "L2")
                                                  .set("dco", "unity")
                                                  .set("num_threads", 12)
                                                  .set("enable_batch_dco", true)
                                                  .create();
  bench(searcher.get(), 0.8, 1000);

  const int k = 10;
  std::vector<int> knn(k);
  searcher->set("ef", 16);
  searcher->ann_search(searcher->get_data(0), k, knn.data());
  for (auto i : knn) {
    std::cout << i << ",";
  }
  std::cout << "\n";
  return 0;
}