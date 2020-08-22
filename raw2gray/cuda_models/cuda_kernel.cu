#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include "cuda_kernel.cuh"

__global__ void raw2gray_kernal(int width, int height, unsigned char *gpu_bayer,
                                unsigned char *gpu_gray) {
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  // G B
  // R G
  float pixel =
      0.7152f * 0.5 *
          (gpu_bayer[2 * index_x + 2 * index_y * (2 * width)] +
           gpu_bayer[2 * index_x + 1 + (2 * index_y + 1) * (2 * width)]) +
      0.2126f * gpu_bayer[2 * index_x + 1 + 2 * index_y * (2 * width)] +
      0.0722f * gpu_bayer[2 * index_x + (2 * index_y + 1) * (2 * width)];

  gpu_gray[index_x + index_y * width] = (unsigned char)pixel;
}

bool cuda_raw2gray(int width, int height, unsigned char *img,
                   unsigned char *res) {
  int BLOCK_SIZE = 24;
  dim3 grid(1);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);

  grid = dim3(((width + (BLOCK_SIZE - 1)) / BLOCK_SIZE),
              ((height + (BLOCK_SIZE - 1)) / BLOCK_SIZE));

  unsigned char *d_img, *d_res;

  cudaMalloc(&d_img, 4 * width * height * sizeof(unsigned char));

  cudaMalloc(&d_res, width * height * sizeof(unsigned char));

  cudaMemcpy(d_img, img, 4 * width * height * sizeof(unsigned char),
             cudaMemcpyHostToDevice);

  raw2gray_kernal<<<grid, block>>>(width, height, d_img, d_res);

  cudaMemcpy(res, d_res, width * height * sizeof(unsigned char),
             cudaMemcpyDeviceToHost);

  cudaFree(d_img);
  cudaFree(d_res);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "raw2gray_kernal failed: %s\n",
            cudaGetErrorString(cudaStatus));
    return false;
  }

  return true;
}

// init cuda enviroment.
int32_t cuda_init(void) {
  int32_t count;
  cudaGetDeviceCount(&count);
  if (count == 0) {
    fprintf(stderr, "There is no device.\n");
    return 0;
  }

  std::cout
      << "-------------------------- GPU dev info --------------------------"
      << std::endl;

  int32_t i;
  for (i = 0; i < count; i++) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      if (prop.major >= 1) {
        printf("Identify: %s\n", prop.name);
        printf("Host Memory: %d\n", (int32_t)prop.canMapHostMemory);
        printf("Clock Rate: %d khz\n", (int32_t)prop.clockRate);
        printf("Compute Mode: %d\n", (int32_t)prop.computeMode);
        printf("Device Overlap: %d\n", (int32_t)prop.deviceOverlap);
        printf("Integrated: %d\n", (int32_t)prop.integrated);
        printf("Kernel Exec Timeout Enabled: %d\n",
               (int32_t)prop.kernelExecTimeoutEnabled);
        printf("Max Grid Size: %d * %d * %d\n", (int32_t)prop.maxGridSize[0],
               (int32_t)prop.maxGridSize[1], (int32_t)prop.maxGridSize[2]);
        printf("Max Threads Dim: %d * %d * %d\n",
               (int32_t)prop.maxThreadsDim[0], (int32_t)prop.maxThreadsDim[1],
               (int32_t)prop.maxThreadsDim[2]);
        printf("Max Threads per Block: %d\n", (int32_t)prop.maxThreadsPerBlock);
        printf("Maximum Pitch: %d bytes\n", (int32_t)prop.memPitch);
        printf("Minor Compute Capability: %d\n", (int32_t)prop.minor);
        printf("Number of Multiprocessors: %d\n",
               (int32_t)prop.multiProcessorCount);
        printf("32bit Registers Availble per Block: %d\n",
               (int32_t)prop.regsPerBlock);
        printf("Shared Memory Available per Block: %d bytes\n",
               (int32_t)prop.sharedMemPerBlock);
        printf("Alignment Requirement for Textures: %d\n",
               (int32_t)prop.textureAlignment);
        printf("Constant Memory Available: %d bytes\n",
               (int32_t)prop.totalConstMem);
        printf("Global Memory Available: %d bytes\n",
               (int32_t)prop.totalGlobalMem);
        printf("Warp Size: %d threads\n", (int32_t)prop.warpSize);
        break;
      }
    }
  }

  if (i == count) {
    fprintf(stderr, "There is no device supporting CUDA.\n");
    return 0;
  }

  cudaSetDevice(i);

  return 1;
}
