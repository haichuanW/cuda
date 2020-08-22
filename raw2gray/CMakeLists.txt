cmake_minimum_required(VERSION 3.1)

project(raw2grayscale)
# local opencv directory
set(OpenCV_DIR /home/haichuan/opencv_build/opencv/build/installation/share/OpenCV)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")
set(CMAKE_BUILD_TYPE Release)

# message("include:" ${OPENCV_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR} ${EIGEN_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR})

add_subdirectory(cuda_models)

include_directories(
    /usr/local/cuda-11.0/include
)

add_executable(raw2gray main.cpp)

target_link_libraries(raw2gray cuda_kernel ${OpenCV_LIBRARIES})

