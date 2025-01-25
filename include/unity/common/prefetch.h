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

// prefetches

#ifdef __AVX__

// SSE

#include <xmmintrin.h>

namespace unity {
inline void prefetch_l1(const void* address) { _mm_prefetch((const char*)address, _MM_HINT_T0); }
inline void prefetch_l2(const void* address) { _mm_prefetch((const char*)address, _MM_HINT_T1); }
inline void prefetch_l3(const void* address) { _mm_prefetch((const char*)address, _MM_HINT_T2); }
}  // namespace unity

#elif defined(__aarch64__)

// ARM64

#ifdef _MSC_VER

// todo: arm on MSVC
namespace unity {
inline void prefetch_l1(const void* address) {}
inline void prefetch_l2(const void* address) {}
inline void prefetch_l3(const void* address) {}
}  // namespace unity

#else
// arm on non-MSVC

namespace unity {
inline void prefetch_l1(const void* address) { __builtin_prefetch(address, 0, 3); }
inline void prefetch_l2(const void* address) { __builtin_prefetch(address, 0, 2); }
inline void prefetch_l3(const void* address) { __builtin_prefetch(address, 0, 1); }
}  // namespace unity
#endif

#else

// a generic platform

#ifdef _MSC_VER

namespace unity {
inline void prefetch_l1(const void* address) {}
inline void prefetch_l2(const void* address) {}
inline void prefetch_l3(const void* address) {}
}  // namespace unity

#else

namespace unity {
inline void prefetch_l1(const void* address) { __builtin_prefetch(address, 0, 3); }
inline void prefetch_l2(const void* address) { __builtin_prefetch(address, 0, 2); }
inline void prefetch_l3(const void* address) { __builtin_prefetch(address, 0, 1); }
}  // namespace unity

#endif

#endif
