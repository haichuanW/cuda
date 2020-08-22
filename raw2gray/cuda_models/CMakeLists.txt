project(cuda_kernel_project)

list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/cmake
)

find_package(CUDALibs)

set(CUDA_NVCC_FLAGS -O3)

cuda_add_library(cuda_kernel SHARED
cuda_kernel.cu
)

target_link_libraries(
cuda_kernel
CUDA::cudart
)