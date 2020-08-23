#include <iostream>
#include "define.h"

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define m 5
#define n 6
#define k 4

void assignValues2Array(float *a, size_t height, size_t width) {
  float ind = width + height;
  for (size_t i = 0; i < width; ++i) {
    for (size_t j = 0; j < height; ++j) {
      a[IDX2C(j, i, height)] = float(ind++);
    }
  }
}

void printArray(float *a, size_t height, size_t width) {
  std::cout << "array: " << std::endl;
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      std::cout << a[IDX2C(i, j, height)] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
  cublasHandle_t handle;

  float *a = (float *)malloc(m * k * sizeof(float));
  float *x = (float *)malloc(k * n * sizeof(float));
  float *y = (float *)malloc(m * n * sizeof(float));

  assignValues2Array(a, m, k);
  assignValues2Array(x, k, n);
  assignValues2Array(y, m, n);

  printArray(a, m, k);
  printArray(x, k, n);
  printArray(y, m, n);

  float *d_x, *d_y, *d_a;

  // k: width(cols), m: height(rows)
  gpuErrchk(cudaMalloc((void **)&d_a, m * k * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&d_x, k * n * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&d_y, m * n * sizeof(float)));

  gpuErrchk(cublasCreate(&handle));

  gpuErrchk(cublasSetMatrix(m, k, sizeof(float), a, m, d_a, m));
  gpuErrchk(cublasSetMatrix(k, n, sizeof(float), x, k, d_x, k));
  gpuErrchk(cublasSetMatrix(m, n, sizeof(float), y, m, d_y, m));

  float alpha = 1.0;
  float beta = 1.0;

  // d_a: m*k matrix, d_x: m*k matrix, d_c: m * n matrix
  // d_y = alpha * d_a * d_x + beta * d_y
  gpuErrchk(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a,
                        m, d_x, k, &beta, d_y, m));

  gpuErrchk(cublasGetMatrix(m, n, sizeof(float), d_y, m, y, m));

  printArray(y, m, n);

  gpuErrchk(cudaFree(d_a));
  gpuErrchk(cudaFree(d_x));
  gpuErrchk(cudaFree(d_y));

  gpuErrchk(cublasDestroy(handle));
  free(a);
  free(x);
  free(y);

  return 0;
}
