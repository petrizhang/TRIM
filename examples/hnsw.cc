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

#include <iostream>

#include "unity/detail/hnswlib/hnswalg.h"
#include "unity/unity.h"

struct RandomGenerator {
  std::mt19937 mt;

  explicit RandomGenerator(int64_t seed = 1234) : mt((unsigned int)seed) {}

  /// random positive integer
  int rand_int() { return mt() & 0x7fffffff; }

  /// random int64_t
  int64_t rand_int64() { return int64_t(rand_int()) | int64_t(rand_int()) << 31; }

  /// generate random integer between 0 and max-1
  int rand_int(int max) { return mt() % max; }

  /// between 0 and 1
  float rand_float() { return mt() / float(mt.max()); }

  double rand_double() { return mt() / double(mt.max()); }
};

std::vector<float> random_matrix(int n_row, int n_col) {
  RandomGenerator gen;
  std::vector<float> x(n_row * n_col);
  for (int i = 0; i < x.size(); i++) {
    x[i] = gen.rand_float();
  }
  return x;
}

int main() {
  int dim = 16;
  int nb = 1000;
  int ef = 64;
  const char* save_path = "./hnswlib.test.bin";

  // Build hnswlib index
  auto space = std::make_unique<hnswlib::L2Space>(dim);
  hnswlib::HierarchicalNSW hnsw(space.get(), nb);
  std::vector<float> base = random_matrix(nb, dim);
  std::cout << "Start to building hnswlib index...\n";
  for (int i = 0; i < nb; i++) {
    hnsw.addPoint(&base.at(i), i);
  }
  hnsw.saveIndex(save_path);
  hnsw.setEf(ef);
  std::cout << "Index saved to " << save_path << "\n";
  auto hnswlib_knn = hnsw.searchKnnCloserFirst(&base.at(0), 10);
  for (auto x : hnswlib_knn) {
    std::cout << x.second << ",";
  }
  std::cout << "\n";

  // Search hnswlib index with top
  std::cout << "Start to load hnswlib index with UNITY...\n";
  std::unique_ptr<unity::Searcher> searcher = unity::SearcherCreator(unity::constants::U_HNSW)
                                                  .set("hnswlib_index_path", save_path)
                                                  .set("dim", dim)
                                                  .set("metric", "L2")
                                                  .set("dco", "exact")
                                                  .create();
  std::cout << "Index loaded.\n";

  constexpr const int k = 10;
  std::vector<int> knn(k);
  const float* data = base.data();
  searcher->set_data(data, nb, dim);
  searcher->set("ef", ef);

  searcher->ann_search(&base.at(0), 10, knn.data());
  for (auto x : knn) {
    std::cout << x << ",";
  }
  std::cout << "\n";

  searcher->ann_search(&base.at(0), 10, knn.data());
  for (auto x : knn) {
    std::cout << x << ",";
  }
  std::cout << "\n";
  return 0;
}