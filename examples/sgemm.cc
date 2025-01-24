#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unity/detail/blas/blas.h"

// Define Fortran-style integer type
typedef int FINTEGER;

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

  float C[M * N] = {0.0f,};

  // Define operation characters
  const char transA = 'N';  // 'N' for no transpose, 'T' for transpose
  const char transB = 'N';  // 'N' for no transpose, 'T' for transpose

  // Call sgemm_
  sgemm_(&transA, &transB, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);

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