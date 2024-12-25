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

#pragma once

#include <chrono>
#include <memory>

#include "top/core/common.h"
#include "top/hnsw/graph.h"
#include "top/hnsw/HNSWInitializer.h"
#include "top/hnswlib/hnswalg.h"
#include "top/hnswlib/hnswlib.h"
#include "top/hnswlib/space_ip.h"
#include "top/hnswlib/space_l2.h"

namespace top {

struct HNSW {
  int nb, dim;
  int M, efConstruction;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw = nullptr;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space = nullptr;

  Graph<int> final_graph;

  HNSW(int dim, int R = 32, int L = 200) : dim(dim), M(R / 2), efConstruction(L) {
    space = std::make_unique<hnswlib::L2Space>(dim);
  }

  Graph<int> GetGraph()  { return final_graph; }
};

}  // namespace top