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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unity/adapter/faiss_blas.h"

int main() {
  // Define matrix dimensions
  FINTEGER M = 3;  // Number of rows in matrix C and A
  FINTEGER N = 3;  // Number of columns in matrix C and B
  FINTEGER K = 3;  // Number of columns in matrix A and rows in matrix B

  // Define leading dimensions
  FINTEGER lda = M;  // Leading dimension of matrix A
  FINTEGER ldb = K;  // Leading dimension of matrix B
  FINTEGER ldc = M;  // Leading dimension of matrix C

  // Define alpha and beta
  float alpha = 1.0f;  // Scaling factor for A * B
  float beta = 0.0f;   // Scaling factor for matrix C

  // Initialize matrices A, B, and C
  float A[M * K] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  float B[K * N] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

  float C[M * N] = {
      0.0f,
  };

  // Define operation characters
  const char transA = 'N';  // 'N' for no transpose, 'T' for transpose
  const char transB = 'N';  // 'N' for no transpose, 'T' for transpose

  // Call sgemm_
  sgemm_faiss(&transA, &transB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

  // Print the result matrix C
  printf("Result matrix C:\n");
  for (FINTEGER i = 0; i < M; ++i) {
    for (FINTEGER j = 0; j < N; ++j) {
      printf("%6.2f ", C[i * N + j]);
    }
    printf("\n");
  }

  return 0;
}