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

#include "unity/detail/io/read_opq.h"

int main() {
  const char* opq_path = "/data/home/petrizhang/develop/TOP/test/index_opq.bin";
  auto opq = unity::detail::read_index_opq(opq_path);
  std::cout << opq->d << "\n";
  std::vector<float> vec(opq->d);
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = i;
  }
  std::cout << "\n";
  std::unique_ptr<const float[]> del(opq->apply_chain(1, vec.data()));
  if (del.get() == vec.data()) {
    del.release();
  }
  
  for (int i = 0; i < vec.size(); i++) {
    std::cout << del[i] << ",";
  }
  std::cout << "\n";
  return 0;
}
