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

#include <cstdio>
#include <cstdlib>
#include <string>

#include "unity/common/u_exception.h"

///
/// Assertions
///

#define U_ASSERT(X)                                         \
  do {                                                      \
    if (!(X)) {                                             \
      fprintf(stderr,                                       \
              "UNITY assertion '%s' failed in %s "          \
              "at %s:%d\n",                                 \
              #X, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
      abort();                                              \
    }                                                       \
  } while (false)

#define U_ASSERT_MSG(X, MSG)                                \
  do {                                                      \
    if (!(X)) {                                             \
      fprintf(stderr,                                       \
              "UNITY assertion '%s' failed in %s "          \
              "at %s:%d; details: " MSG "\n",               \
              #X, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
      abort();                                              \
    }                                                       \
  } while (false)

#define U_ASSERT_FMT(X, FMT, ...)                                        \
  do {                                                                   \
    if (!(X)) {                                                          \
      fprintf(stderr,                                                    \
              "UNITY assertion '%s' failed in %s "                       \
              "at %s:%d; details: " FMT "\n",                            \
              #X, __PRETTY_FUNCTION__, __FILE__, __LINE__, __VA_ARGS__); \
      abort();                                                           \
    }                                                                    \
  } while (false)

///
/// Exceptions for returning user errors
///

#define U_THROW_MSG(MSG)                                                       \
  do {                                                                         \
    throw unity::UnityException(MSG, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
  } while (false)

#define U_THROW_FMT(FMT, ...)                                                  \
  do {                                                                         \
    std::string __s;                                                           \
    int __size = snprintf(nullptr, 0, FMT, __VA_ARGS__);                       \
    __s.resize(__size + 1);                                                    \
    snprintf(&__s[0], __s.size(), FMT, __VA_ARGS__);                           \
    throw unity::UnityException(__s, __PRETTY_FUNCTION__, __FILE__, __LINE__); \
  } while (false)

///
/// Exceptions thrown upon a conditional failure
///

#define U_THROW_IF_NOT(X)                    \
  do {                                       \
    if (!(X)) {                              \
      U_THROW_FMT("Error: '%s' failed", #X); \
    }                                        \
  } while (false)

#define U_THROW_IF_MSG(X, MSG)                     \
  do {                                             \
    if (X) {                                       \
      U_THROW_FMT("Error: '%s' failed: " MSG, #X); \
    }                                              \
  } while (false)

#define U_THROW_IF_NOT_MSG(X, MSG) U_THROW_IF_MSG(!(X), MSG)

#define U_THROW_IF_NOT_FMT(X, FMT, ...)                         \
  do {                                                          \
    if (!(X)) {                                                 \
      U_THROW_FMT("Error: '%s' failed: " FMT, #X, __VA_ARGS__); \
    }                                                           \
  } while (false)

#define U_THROW_NOT_IMPLEMENTED U_THROW_MSG("not implemented error")