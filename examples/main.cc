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

#include "top/factory.h"

int main() {
  using top::Dict;
  using top::Object;
  using top::Searcher;
  using top::SearcherBuilder;

  std::unique_ptr<Searcher> searcher =
      SearcherBuilder(top::constants::TOP_HNSW)
          .set(top::constants::TOP_HNSWLIB_INDEX_PATH,
               "/data/home/petrizhang/develop/TOP/examples/hnswlib.bin")
          .set(top::constants::TOP_DIM, 256)
          .set(top::constants::TOP_METRIC, top::constants::TOP_METRIC_L2)
          .build();

  std::cout << searcher->get_profile().to_string() << "\n";
  return 0;
}