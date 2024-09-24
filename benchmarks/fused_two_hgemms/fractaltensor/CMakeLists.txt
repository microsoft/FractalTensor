cmake_minimum_required(VERSION 3.18)
project(fused_two_hgemms CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
            "${CMAKE_SOURCE_DIR}/../../../cmake/Modules/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

find_package(CUDA QUIET REQUIRED)
find_package(CuDNN QUIET REQUIRED)

cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")
message(STATUS "CUDA Architecture flags = ${ARCH_FLAGS}")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(CUTLASS_NATIVE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --expt-relaxed-constexpr)
else()
  list(APPEND CUTLASS_CUDA_NVCC_FLAGS --std=c++17)
endif()

set(CUDA_NVCC_FLAGS ${CUTLASS_CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS} -w
                    ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_DEBUG ${CUTLASS_CUDA_NVCC_FLAGS} ${CUDA_NVCC_FLAGS_DEBUG}
                          -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_RELEASE ${CUTLASS_CUDA_NVCC_FLAGS}
                            ${CUDA_NVCC_FLAGS_RELEASE} -w -O3 ${ARCH_FLAGS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS_DEBUG
    "$ENV{CXXFLAGS} -O0 -fPIC -Wall -Wno-sign-compare -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -fPIC -O3 -Wall
    -Wno-sign-compare")
set(CMAKE_CXX_LINK_EXECUTABLE
    "${CMAKE_CXX_LINK_EXECUTABLE} -lpthread -ldl -lrt")

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIRS})

# FIXME(ying): this requires to build the main project first. And it is a quite
# a tricky way to build the benchmark.
include_directories(
  "../../../build/third_party/cutlass/src/extern_cutlass/include")
include_directories(
  "../../../build/third_party/cutlass/src/extern_cutlass/tools/util/include")
include_directories("../../../build/third_party/install/glog/include")
include_directories(
  "../../../build/third_party/gflags/src/extern_gflags-build/include")
include_directories("../../../")
link_directories("../../../build/kaleido/core")
link_directories("../../../build/kaleido/core/operators")

cuda_add_executable(hgemm_b2b fused_two_hgemms.cu)
target_link_libraries(
  hgemm_b2b
  fill_op
  print_op
  expect_eq_op
  fractaltensor_core
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
  ${CUDNN_LIBRARIES})
