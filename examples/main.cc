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

#include "top/common/dict.h"
#include "top/common/object.h"
#include "top/common/searcher.h"
#include "top/detail/io/read_faiss.h"
#include "top/detail/io/read_hnswlib.h"
#include "top/detail/quant/pq.h"
#include "top/detail/searcher/hnsw_searcher.h"

int main() {
  using top::Dict;
  using top::Object;
  using top::detail::Graph;
  using top::detail::Index;
  using top::detail::IndexPQ;
  using top::detail::IndexType;

  const char* index_pa_path = "/data/home/petrizhang/develop/TOP/examples/index_pq.bin";
  const char* hnswlib_path = "/data/home/petrizhang/develop/TOP/examples/hnswlib.bin";
  Graph<int> graph = top::detail::read_hnswlib(hnswlib_path, 256);

  std::unique_ptr<IndexPQ> index_pq = top::detail::read_index_pq(index_pa_path);

  std::cout << reinterpret_cast<int64_t>(index_pq.get()) << "\n";

  Object x(true), y(1.3123), z(6), q("this is a string");
  std::cout << "x: " << x.to_string() << "\n";
  std::cout << "y: " << y.to_string() << "\n";
  std::cout << "z: " << z.to_string() << "\n";
  std::cout << "q: " << q.to_string() << "\n";

  Dict dict;
  dict.put("test", "a");
  dict.put("ef", 1);
  dict.put("gamma", 3.1415926535459);
  dict.put("use_bounded_queue", false);
  std::cout << dict.to_string() << "\n";
  return 0;
}