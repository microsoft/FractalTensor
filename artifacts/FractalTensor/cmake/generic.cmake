set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS_DEBUG
    "$ENV{CXXFLAGS} -O0 -fPIC -Wall -Wno-sign-compare -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -fPIC -O3 -Wall -Wno-sign-compare")

set(CMAKE_CXX_LINK_EXECUTABLE
    "${CMAKE_CXX_LINK_EXECUTABLE} -lpthread -ldl -lrt")

cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${ARCH_FLAGS}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
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

if(CUDA_VERSION VERSION_LESS 11.3)
  message(
    WARNING
      "CUTLASS ${CUTLASS_VERSION} requires CUDA 11.4 or higher, and strongly recommends CUDA 11.8 or higher."
  )
elseif(CUDA_VERSION VERSION_LESS 11.4)
  message(
    WARNING
      "CUTLASS ${CUTLASS_VERSION} support for CUDA ${CUDA_VERSION} is deprecated, please use CUDA 11.8 or higher."
  )
endif()

function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  if(cc_library_SRCS)
    if(cc_library_SHARED) # build *.so
      add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
    endif()

    if(cc_library_DEPS)
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    endif()

    # cpplint code style
    foreach(source_file ${cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cc_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()
  else(cc_library_SRCS)
    if(cc_library_DEPS)
      list(REMOVE_DUPLICATES cc_library_DEPS)
      set(target_SRCS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)
      file(WRITE ${target_SRCS}
           "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")

      add_library(${TARGET_NAME} STATIC ${target_SRCS})
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL_ERROR "No source file is given.")
    endif()
  endif(cc_library_SRCS)
endfunction(cc_library)

function(cc_test_build TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  add_executable(
    ${TARGET_NAME} ${PROJECT_SOURCE_DIR}/kaleido/core/tests/test_main.cc
                   ${cc_test_SRCS})
  target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} gtest
                        ${CUDA_curand_LIBRARY})
endfunction()

function(nv_test TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(nv_test "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  cuda_add_executable(
    ${TARGET_NAME} ${PROJECT_SOURCE_DIR}/kaleido/core/tests/test_main.cc
    ${nv_test_SRCS})
  target_link_libraries(${TARGET_NAME} ${nv_test_DEPS} gtest glog gflags)
  add_dependencies(${TARGET_NAME} ${nv_test_DEPS} gtest glog gflags)
  add_test(${TARGET_NAME} ${TARGET_NAME})
endfunction(nv_test)

function(cpp_proto_generate TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS)
  cmake_parse_arguments(cpp_proto_generate "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  set(proto_srcs)
  set(proto_hdrs)
  protobuf_generate_cpp(proto_srcs proto_hdrs ${cpp_proto_generate_SRCS})
  cc_library(${TARGET_NAME} SRCS ${proto_srcs} DEPS ${proto_library_DEPS})
endfunction()

function(py_proto_generate TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS)
  cmake_parse_arguments(py_proto_generate "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  set(py_srcs)
  protobuf_generate_python(py_srcs ${py_proto_generate_SRCS})
  add_custom_target(${TARGET_NAME} ALL DEPENDS ${py_srcs})
endfunction()

function(nv_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(nv_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  if(nv_library_SHARED OR nv_library_shared) # build *.so
    cuda_add_library(${TARGET_NAME} SHARED ${nv_library_SRCS})
  else()
    cuda_add_library(${TARGET_NAME} STATIC ${nv_library_SRCS})
  endif()
  if(nv_library_DEPS)
    add_dependencies(${TARGET_NAME} ${nv_library_DEPS})
    target_link_libraries(${TARGET_NAME} ${nv_library_DEPS})
  endif()
endfunction(nv_library)

function(op_library TARGET)
  set(cc_srcs)
  set(cu_srcs)
  set(multiValueArgs DEPS)
  cmake_parse_arguments(op_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cc)
    list(APPEND cc_srcs ${TARGET}.cc)
  endif()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TARGET}.cu)
    list(APPEND cu_srcs ${TARGET}.cu)
  endif()

  list(LENGTH cc_srcs cc_srcs_len)
  list(LENGTH cu_srcs cu_srcs_len)

  # separate the compilation for *.cu and *.cc
  if(${cu_srcs_len} GREATER 0)
    nv_library(${TARGET} SRCS ${cc_srcs} ${cu_srcs})
    target_link_libraries(${TARGET} ${op_library_DEPS})
  endif()
  if(${cc_srcs_len} GREATER 0)
    cc_library(${TARGET} SRCS ${cc_srcs} DEPS ${op_library_DEPS})
  endif()
endfunction()
