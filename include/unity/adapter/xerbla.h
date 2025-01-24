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

#include <stdio.h>

#if (defined __GNUC__) && (!defined __MINGW32__) && (!defined __CYGWIN__)
#define EIGEN_WEAK_LINKING __attribute__((weak))
#else
#define EIGEN_WEAK_LINKING
#endif

#ifdef __cplusplus
extern "C" {
#endif

EIGEN_WEAK_LINKING int xerbla_(const char* msg, int* info, int) {
  printf("Eigen BLAS ERROR #%i: %s\n", *info, msg);
  return 0;
}

#ifdef __cplusplus
}
#endif
