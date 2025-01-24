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

#include "./blas.h"
#include "unity/common/u_assert.h"

#ifdef FINTEGER
#undef FINTEGER
#endif
#define FINTEGER int

#define Run_get_distance_computer Run_get_distance_computer_extra
#include "faiss/utils/extra_distances.cpp"
#undef Run_get_distance_computer

#define sgemv_ sgemv_faiss
#define sgemm_ sgemm_faiss
#define dgemm_ dgemm_faiss
#define ssyrk_ ssyrk_faiss
#define ssyev_ ssyev_faiss
#define dsyev_ dsyev_faiss
#define sgesvd_ sgesvd_faiss
#define dgesvd_ dgesvd_faiss
#define sgeqrf_ sgeqrf_faiss
#define sorgqr_ sorgqr_faiss

#include "faiss/Clustering.cpp"
#include "faiss/Index.cpp"
#include "faiss/IndexFlat.cpp"
#include "faiss/IndexFlatCodes.cpp"
#include "faiss/IndexIVF.cpp"
#include "faiss/IndexIVFPQ.cpp"
#include "faiss/IndexPQ.cpp"
#include "faiss/VectorTransform.cpp"
#include "faiss/impl/AuxIndexStructures.cpp"
#include "faiss/impl/CodePacker.cpp"
#include "faiss/impl/FaissException.cpp"
#include "faiss/impl/IDSelector.cpp"
#include "faiss/impl/PolysemousTraining.cpp"
#include "faiss/impl/ProductQuantizer.cpp"
#include "faiss/impl/io.cpp"
#include "faiss/impl/kmeans1d.cpp"
#include "faiss/invlists/BlockInvertedLists.cpp"
#include "faiss/invlists/DirectMap.cpp"
#include "faiss/invlists/InvertedLists.cpp"
#include "faiss/invlists/InvertedListsIOHook.cpp"
#include "faiss/invlists/OnDiskInvertedLists.cpp"
#include "faiss/utils/Heap.cpp"
#include "faiss/utils/distances.cpp"
#include "faiss/utils/distances_fused/distances_fused.cpp"
#include "faiss/utils/distances_fused/simdlib_based.cpp"
#include "faiss/utils/distances_simd.cpp"
#include "faiss/utils/hamming.cpp"
#include "faiss/utils/partitioning.cpp"
#include "faiss/utils/random.cpp"
#include "faiss/utils/sorting.cpp"
#include "faiss/utils/utils.cpp"
#include "unity/adapter/fake_omp.h"

#undef sgemv_
#undef sgemm_
#undef dgemm_
#undef ssyrk_
#undef ssyev_
#undef dsyev_
#undef sgesvd_
#undef dgesvd_
#undef sgeqrf_
#undef sorgqr_

extern "C" {
int sgemv_faiss(const char* trans, FINTEGER* m, FINTEGER* n, float* alpha, const float* a,
                FINTEGER* lda, const float* x, FINTEGER* incx, float* beta, float* y,
                FINTEGER* incy) {
  return sgemv_eigen(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

int sgemm_faiss(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n, FINTEGER* k,
                const float* alpha, const float* a, FINTEGER* lda, const float* b, FINTEGER* ldb,
                float* beta, float* c, FINTEGER* ldc) {
  return sgemm_eigen(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int dgemm_faiss(const char* transa, const char* transb, FINTEGER* m, FINTEGER* n, FINTEGER* k,
                const double* alpha, const double* a, FINTEGER* lda, const double* b, FINTEGER* ldb,
                double* beta, double* c, FINTEGER* ldc) {
  return dgemm_eigen(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int ssyrk_faiss(const char* uplo, const char* trans, FINTEGER* n, FINTEGER* k, float* alpha,
                float* a, FINTEGER* lda, float* beta, float* c, FINTEGER* ldc) {
  return ssyrk_eigen(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

int ssyev_faiss(const char* jobz, const char* uplo, FINTEGER* n, float* a, FINTEGER* lda, float* w,
                float* work, FINTEGER* lwork, FINTEGER* info) {
  U_THROW_NOT_IMPLEMENTED;
  return 0;
}

int dsyev_faiss(const char* jobz, const char* uplo, FINTEGER* n, double* a, FINTEGER* lda,
                double* w, double* work, FINTEGER* lwork, FINTEGER* info) {
  U_THROW_NOT_IMPLEMENTED;
  return 0;
}

int sgesvd_faiss(const char* jobu, const char* jobvt, FINTEGER* m, FINTEGER* n, float* a,
                 FINTEGER* lda, float* s, float* u, FINTEGER* ldu, float* vt, FINTEGER* ldvt,
                 float* work, FINTEGER* lwork, FINTEGER* info) {
  U_THROW_NOT_IMPLEMENTED;
  return 0;
}

int dgesvd_faiss(const char* jobu, const char* jobvt, FINTEGER* m, FINTEGER* n, double* a,
                 FINTEGER* lda, double* s, double* u, FINTEGER* ldu, double* vt, FINTEGER* ldvt,
                 double* work, FINTEGER* lwork, FINTEGER* info) {
  U_THROW_NOT_IMPLEMENTED;
  return 0;
}

int sgeqrf_faiss(FINTEGER* m, FINTEGER* n, float* a, FINTEGER* lda, float* tau, float* work,
                 FINTEGER* lwork, FINTEGER* info) {
  U_THROW_NOT_IMPLEMENTED;
  return 0;
}

int sorgqr_faiss(FINTEGER* m, FINTEGER* n, FINTEGER* k, float* a, FINTEGER* lda, float* tau,
                 float* work, FINTEGER* lwork, FINTEGER* info) {
  U_THROW_NOT_IMPLEMENTED;
  return 0;
}
}