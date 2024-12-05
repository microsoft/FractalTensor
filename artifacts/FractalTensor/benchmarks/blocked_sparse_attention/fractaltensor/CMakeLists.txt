cmake_minimum_required(VERSION 3.18)

project(bigbird CXX C)

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS_DEBUG
    "$ENV{CXXFLAGS} -O0 -fPIC -Wall -Wno-sign-compare -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -fPIC -O3 -Wall
    -Wno-sign-compare")

cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")
message(STATUS "CUDA Architecture flags = ${ARCH_FLAGS}")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CMAKE_CXX_LINK_EXECUTABLE
    "${CMAKE_CXX_LINK_EXECUTABLE} -lpthread -ldl -lrt")

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIRS})
include_directories("../../../") # include the project's core directory

cuda_add_executable(bigbird bigbird_bench.cu)
target_link_libraries(bigbird ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}
                      ${CUDA_curand_LIBRARY} ${CUDNN_LIBRARIES})
