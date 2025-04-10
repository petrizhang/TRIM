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

#ifdef EIGEN_BLAS_FUNC_SUFFIX
#undef EIGEN_BLAS_FUNC_SUFFIX
#endif
#define EIGEN_BLAS_FUNC_SUFFIX _eigen

#include "trim/adapter/xerbla.h"

//////////////////////////////////////////////////////
// Single precision BLAS functions
//////////////////////////////////////////////////////

#undef SCALAR
#undef SCALAR_SUFFIX
#undef SCALAR_SUFFIX_UP
#undef ISCOMPLEX

#define SCALAR float
#define SCALAR_SUFFIX s
#define SCALAR_SUFFIX_UP "S"
#define ISCOMPLEX 0

#ifdef EIGEN_BLAS_FUNC
#undef EIGEN_BLAS_FUNC
#endif
#define EIGEN_BLAS_FUNC(X) EIGEN_CAT(SCALAR_SUFFIX, EIGEN_CAT(X, EIGEN_BLAS_FUNC_SUFFIX))

#include "blas/level1_impl.h"
#include "blas/level1_real_impl.h"
#include "blas/level2_impl.h"
#include "blas/level2_real_impl.h"
#include "blas/level3_impl.h"

// WARNING: sdsdot_ is not supported now
// float EIGEN_BLAS_FUNC(dsdot)(int* n, float* alpha, float* x, int* incx, float* y, int* incy) {}

#undef SCALAR
#undef SCALAR_SUFFIX
#undef SCALAR_SUFFIX_UP
#undef ISCOMPLEX

#undef EIGEN_BLAS_FUNC