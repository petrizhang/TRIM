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

#include <string>

#ifdef LIKELY
#undef LIKELY
#endif

#ifdef UNLIKELY
#undef UNLIKELY
#endif

#define LIKELY(expr) __builtin_expect(!!(expr), 1)
#define UNLIKELY(expr) __builtin_expect(!!(expr), 0)

#define PREFETCH(addr) __builtin_prefetch(addr)

/// Force inlining. The 'inline' keyword is treated by most compilers as a hint,
/// not a command. This should be used sparingly for cases when either the function
/// needs to be inlined for a specific reason or the compiler's heuristics make a bad
/// decision, e.g. not inlining a small function on a hot path.
#define ALWAYS_INLINE __attribute__((always_inline))

#define U_FORBID_DEFAULT_CTOR(TypeName) TypeName() = delete;

#define U_FORBID_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  TypeName& operator=(const TypeName&) = delete;

#define U_FORBID_MOVE(TypeName)  \
  TypeName(TypeName&&) = delete; \
  TypeName& operator=(TypeName&&) = delete;

namespace trim {

enum class Metric {
  L2,
  IP,
};

inline constexpr size_t upper_div(size_t x, size_t y) { return (x + y - 1) / y; }

inline constexpr int64_t do_align(int64_t x, int64_t align) {
  return (x + align - 1) / align * align;
}

#define FAST_BEGIN            \
  _Pragma("GCC push_options") \
      _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")

#define FAST_END _Pragma("GCC pop_options")

}  // namespace trim