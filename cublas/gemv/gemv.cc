#include <cuda_runtime.h>

#include <iostream>
#include "define.h"

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define m 5
#define n 6

int main(int argc, char **argv) {
  cublasHandle_t handle;

  float *a = (float *)malloc(m * n * sizeof(float));
  float *x = (float *)malloc(n * sizeof(float));
  float *y = (float *)malloc(m * sizeof(float));

  for (size_t i = 0; i < m * n; ++i) {
    a[i] = 1.0;
  }

  for (size_t i = 0; i < n; ++i) {
    x[i] = 1.0;
  }

  for (size_t i = 0; i < m; ++i) {
    y[i] = 1.0;
  }

  float *d_x, *d_y, *d_a;

  gpuErrchk(cudaMalloc((void **)&d_a, m * n * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&d_x, n * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&d_y, m * sizeof(float)));

  gpuErrchk(cublasCreate(&handle));

  gpuErrchk(cublasSetMatrix(m, n, sizeof(float), a, m, d_a, m));
  gpuErrchk(cublasSetVector(n, sizeof(float), x, 1, d_x, 1));
  gpuErrchk(cublasSetVector(m, sizeof(float), y, 1, d_y, 1));

  float alpha = 1.0;
  float beta = 1.0;

  // d_y = alpha * d_a * d_x + beta * d_y
  //   d_a: m * n matrix, d_x: n-vector, d_y: m-vector
  gpuErrchk(cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_a, m, d_x, 1,
                        &beta, d_y, 1));

  gpuErrchk(cublasGetVector(m, sizeof(float), d_y, 1, y, 1));

  for (size_t i = 0; i < m; ++i) {
    std::cout << y[i] << std::endl;
  }

  gpuErrchk(cudaFree(d_a));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_y));

  gpuErrchk(cublasDestroy(handle));
  free(a);
  free(x);
  free(y);

  return 0;
}
