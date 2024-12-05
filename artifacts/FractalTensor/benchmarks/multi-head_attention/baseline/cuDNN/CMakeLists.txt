cmake_minimum_required(VERSION 3.0)
project(benchmarks CXX C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
            "${CMAKE_SOURCE_DIR}/../../../../../cmake/Modules/")

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
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -w ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -w -O3 ${ARCH_FLAGS})

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS_DEBUG
    "$ENV{CXXFLAGS} -O0 -fPIC -Wall -Wno-sign-compare -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE
    "$ENV{CXXFLAGS} -fPIC -O3 -Werror -Wno-sign-compare")
set(CMAKE_CXX_LINK_EXECUTABLE
    "${CMAKE_CXX_LINK_EXECUTABLE} -lpthread -ldl -lrt")

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIRS})

# cuda_add_executable(cudnn_mha multiHeadAttention.cpp)
cuda_add_executable(cudnn_mha main.cu)
target_link_libraries(cudnn_mha ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY}
                      ${CUDNN_LIBRARIES})
