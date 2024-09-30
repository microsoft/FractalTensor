cmake_minimum_required(VERSION 3.18) # cutlass 3.2 requires cmake 3.18+

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
            "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(PYPARSER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PYPARSER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PYPARSER_PYTHON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/build")

project(kaleido CXX C)
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
find_package(CuDNN REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS filesystem regex)

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIR})
  add_definitions("-DHAS_BOOST")
else()
  message(FATAL_ERROR "Cannot find Boost.")
endif()

include(generic)
include(python)
include(third_party)

add_subdirectory(kaleido/core)
