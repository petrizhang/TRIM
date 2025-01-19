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

#include <atomic>
#include <utility>

namespace unity {
template <typename T>
struct Atomic {
  std::atomic<T> value;

  Atomic() = default;

  explicit Atomic(T init_value) : value(init_value) {}

  Atomic(const Atomic& other) : value(other.value.load()) {}

  Atomic& operator=(const Atomic& other) {
    if (this != &other) {
      value.store(other.value.load());
    }
    return *this;
  }

  Atomic(Atomic&& other) noexcept : value(other.value.load()) {}

  Atomic& operator=(Atomic&& other) noexcept {
    if (this != &other) {
      value.store(other.value.load());
    }
    return *this;
  }

  void swap(Atomic& other) noexcept {
    if (this != &other) {
      T temp = value.load();
      value.store(other.value.load());
      other.value.store(temp);
    }
  }
};

}  // namespace unity