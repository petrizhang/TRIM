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

#include "unity/detail/hnsw/top_deo.h"
#include "unity/detail/io/read_faiss.h"
#include "unity/detail/quantization/index_pq.h"
#include "unity/unity.h"

int main() {
  using unity::Searcher;
  using unity::detail::Index;
  using unity::detail::IndexPQ;
  using unity::detail::IndexType;

  // Search hnswlib index with top
  const char* index_hnsw_path = "/data/home/petrizhang/develop/TOP/examples/hnswlib.bin";
  const char* index_pq_path = "/data/home/petrizhang/develop/TOP/examples/index_pq.bin";
  const int dim = 256;
  std::unique_ptr<unity::Searcher> searcher = unity::SearcherCreator(unity::constants::U_HNSW)
                                                .set("hnswlib_index_path", index_hnsw_path)
                                                .set("pq_index_path", index_pq_path)
                                                .set("dim", dim)
                                                .set("metric", "L2")
                                                .set("num_threads", 12)
                                                .build();
  std::cout << (int64_t)searcher.get() << "\n";
  return 0;
}