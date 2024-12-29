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

#include "top/detail/io/read_faiss.h"
#include "top/detail/quantization/index_pq.h"

int main() {
  using top::detail::Index;
  using top::detail::IndexPQ;
  using top::detail::IndexType;
  const char* index_path = "/data/home/petrizhang/develop/TOP/examples/index_pq.bin";
  std::unique_ptr<IndexPQ> index_pq = top::detail::read_index_pq(index_path);
  std::vector<float> data(1000 * 256);
  for (int i = 0; i < data.size(); i++) {
    data[i] = 0;
  }

  ctpl::thread_pool pool(10);
  index_pq->compute_reconstruction_errors(pool, data.data());
  std::cout << (int64_t)index_pq.get() << "\n";
  return 0;
}