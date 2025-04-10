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

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
// #include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <random>

#include "trim/detail/io/read_faiss.h"

int main() {
  // dimension of the vectors to index
  int d = 64;

  // size of the database we plan to index
  size_t nb = 1000;

  // make a set of nt training vectors in the unit cube
  // (could be the database)
  size_t nt = 1500;

  // make the index object and train it
  faiss::IndexFlatL2 coarse_quantizer(d);

  // a reasonable number of cetroids to index nb vectors
  int ncentroids = 25;

  faiss::IndexIVFPQ index(&coarse_quantizer, d, ncentroids, 16, 8);

  // index that gives the ground-truth
  faiss::IndexFlatL2 index_gt(d);

  std::mt19937 rng;
  std::uniform_real_distribution<> distrib;

  {  // training

    std::vector<float> trainvecs(nt * d);
    for (size_t i = 0; i < nt * d; i++) {
      trainvecs[i] = distrib(rng);
    }
    index.verbose = true;
    index.train(nt, trainvecs.data());
  }

  {  // populating the database

    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
      database[i] = distrib(rng);
    }

    index.add(nb, database.data());
    index_gt.add(nb, database.data());
  }

  int nq = 200;
  int n_ok;

  {  // searching the database

    std::vector<float> queries(nq * d);
    for (size_t i = 0; i < nq * d; i++) {
      queries[i] = distrib(rng);
    }

    std::vector<faiss::idx_t> gt_nns(nq);
    std::vector<float> gt_dis(nq);

    index_gt.search(nq, queries.data(), 1, gt_dis.data(), gt_nns.data());

    index.nprobe = 5;
    int k = 5;
    std::vector<faiss::idx_t> nns(k * nq);
    std::vector<float> dis(k * nq);

    index.search(nq, queries.data(), k, dis.data(), nns.data());

    n_ok = 0;
    for (int q = 0; q < nq; q++) {
      for (int i = 0; i < k; i++)
        if (nns[q * k + i] == gt_nns[q]) n_ok++;
    }
    assert(n_ok = nq * 0.4);
  }
}