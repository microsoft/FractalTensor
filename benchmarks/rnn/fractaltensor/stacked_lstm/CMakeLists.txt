cmake_minimum_required(VERSION 3.18)
project(recurrence_test CXX C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
            "${CMAKE_SOURCE_DIR}/../../../../cmake/Modules/")

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS_DEBUG
    "$ENV{CXXFLAGS} -O0 -fPIC -Wall -Wno-sign-compare -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -fPIC -O3 -Wall -Wno-sign-compare")

set(CMAKE_CXX_LINK_EXECUTABLE
    "${CMAKE_CXX_LINK_EXECUTABLE} -lpthread -ldl -lrt")

find_package(CUDA QUIET REQUIRED)
find_package(CuDNN QUIET REQUIRED)

cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")
message(STATUS "CUDA Architecture flags = ${ARCH_FLAGS}")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# FIXME(ying): The RNN examples do not rely on cutlass, but the fill kernel in
# `fill.h` depends on cutlass, and cutlass require C++17. This is a hotfix to
# bypass the compiling error. Make the dependency clean in the future.
if(CUTLASS_NATIVE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --expt-relaxed-constexpr)
else()
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --std=c++17)
endif()
include_directories(
  "../../../../build/third_party/cutlass/src/extern_cutlass/include")
include_directories(
  "../../../../build/third_party/cutlass/src/extern_cutlass/tools/util/include")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -w -O3 ${ARCH_FLAGS})

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIRS})

include_directories("../")
include_directories("../../../../")
include_directories("../../../../build/third_party/install/glog/include")
include_directories(
  "../../../../build/third_party/gflags/src/extern_gflags-build/include")
link_directories("../../../../build/kaleido/core")
link_directories("../../../../build/kaleido/core/operators")

cuda_add_executable(lstm lstm.cu)
target_link_libraries(
  lstm
  ${CUDA_LIBRARIES}
  ${CUDNN_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  fractaltensor_core
  print_op)
