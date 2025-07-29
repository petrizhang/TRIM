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

#include <fstream>
#include <iostream>
#include <vector>

#include "trim/detail/index/tIVFPQfs.h"

// 从二进制文件加载矩阵
std::vector<float> load_matrix(const std::string& dataPath, size_t rows, size_t cols) {
  // 读取矩阵数据
  std::ifstream dataFile(dataPath, std::ios::binary);
  if (!dataFile.is_open()) {
    throw std::runtime_error("无法打开数据文件");
  }

  // 计算总元素数
  size_t totalElements = rows * cols;
  std::vector<float> flatData(totalElements);

  // 读取所有数据到一维数组
  dataFile.read(reinterpret_cast<char*>(flatData.data()), totalElements * sizeof(float));
  dataFile.close();

  return flatData;
}

uint32_t get_lt_mask(float* a, float* b) {
  __m512 va = _mm512_loadu_ps(a);
  __m512 vb = _mm512_loadu_ps(b);
  auto mask = _mm512_cmp_ps_mask(va, vb, _CMP_LT_OQ);
  return mask;
}

void print_mask(uint32_t mask, const char* name) {
  std::cout << name << ": ";
  while (mask) {
    // find first non-zero
    int j = __builtin_ctz(mask);
    mask -= 1 << j;
    std::cout << j << ",";
  }
  std::cout << "\n";
}

int main() {
  std::vector<float> a0 = {1, 0, 3, 4, 5, 6, 7, 8, 0, 10, 11, 12, 13, 14, 15, 16};
  std::vector<float> b0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  std::vector<float> a1 = {1, 2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0};
  std::vector<float> b1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

  auto mask0 = get_lt_mask(a0.data(), b0.data());
  auto mask1 = get_lt_mask(a1.data(), b1.data());
  print_mask(mask0, "mask0");
  print_mask(mask1, "mask1");

  __mmask32 lt_mask = _kor_mask32(_kshiftli_mask32(mask1, 16), mask0);
  print_mask(lt_mask, "lt_mask");

  using namespace faiss;
  const char* index_path = "/home/yitong/petrizhang/TOP/index_ivfpqfs.bin";
  const char* data_path = "/home/yitong/petrizhang/TOP/data.bin";
  auto vector_data = load_matrix(data_path, 1000, 256);

  tIVFPQfs index(index_path);
  index.set_data(vector_data.data());
  index.compute_recons_errors();

  int k = 10;
  std::vector<int64_t> ids(k);

  std::vector<float> distances(k);
  index.search(1, vector_data.data(), k, distances.data(), ids.data());
  for (auto f : distances) {
    std::cout << f << ",";
  }
  std::cout << "\n";

  for (auto id : ids) {
    std::cout << id << ",";
  }
  std::cout << "\n";
  std::cout << "Index Read: ntotal=" << index.ntotal << ", nlist=" << index.nlist
            << ", M=" << index.M << ", nbtis=" << index.nbits << ", bbs=" << index.bbs << "\n";
  std::cout << index._pruning_ratio;
  std::cout << std::endl;
  return 0;
}
