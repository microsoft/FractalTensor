cmake_minimum_required(VERSION 3.0)
project(tvm_test CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS_DEBUG
    "$ENV{CXXFLAGS} -O0 -fPIC -Wall -Wno-sign-compare -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -fPIC -O3 -Wall -Wno-sign-compare")

find_package(CUDA QUIET REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(TVM_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../build/third_party/tvm/src/extern_tvm/"
)

include_directories(${TVM_ROOT}/include ${TVM_ROOT}/3rdparty/dmlc-core/include
                    ${TVM_ROOT}/3rdparty/dlpack/include)
link_directories(${TVM_ROOT})

add_executable(main main.cc)
target_link_libraries(
  main
  tvm_runtime
  cuda
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY})
