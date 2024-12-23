# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

include(ExternalProject)

find_package(Protobuf QUIET)

unset(PROTOBUF_INCLUDE_DIR)
unset(PROTOBUF_FOUND)
unset(PROTOBUF_PROTOC_EXECUTABLE)
unset(PROTOBUF_PROTOC_LIBRARY)
unset(PROTOBUF_LITE_LIBRARY)
unset(PROTOBUF_LIBRARY)
unset(PROTOBUF_INCLUDE_DIR)
unset(Protobuf_PROTOC_EXECUTABLE)

set(PROTOBUF_VERSION 3.6.1)

function(build_protobuf TARGET_NAME BUILD_FOR_HOST)
  string(REPLACE "extern_" "" TARGET_DIR_NAME "${TARGET_NAME}")
  set(PROTOBUF_PREFIX_DIR ${THIRD_PARTY_PATH}/${TARGET_DIR_NAME})
  set(PROTOBUF_SOURCE_DIR
      ${THIRD_PARTY_PATH}/${TARGET_DIR_NAME}/src/${TARGET_NAME})
  set(PROTOBUF_INSTALL_DIR ${THIRD_PARTY_PATH}/install/${TARGET_DIR_NAME})

  set(${TARGET_NAME}_INCLUDE_DIR
      "${PROTOBUF_INSTALL_DIR}/include"
      PARENT_SCOPE)
  set(PROTOBUF_INCLUDE_DIR
      "${PROTOBUF_INSTALL_DIR}/include"
      PARENT_SCOPE)
  set(${TARGET_NAME}_LITE_LIBRARY
      "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite${CMAKE_STATIC_LIBRARY_SUFFIX}"
      PARENT_SCOPE)
  set(${TARGET_NAME}_LIBRARY
      "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf${CMAKE_STATIC_LIBRARY_SUFFIX}"
      PARENT_SCOPE)
  set(${TARGET_NAME}_PROTOC_LIBRARY
      "${PROTOBUF_INSTALL_DIR}/lib/libprotoc${CMAKE_STATIC_LIBRARY_SUFFIX}"
      PARENT_SCOPE)
  set(${TARGET_NAME}_PROTOC_EXECUTABLE
      "${PROTOBUF_INSTALL_DIR}/bin/protoc${CMAKE_EXECUTABLE_SUFFIX}"
      PARENT_SCOPE)

  set(OPTIONAL_CACHE_ARGS "")
  set(OPTIONAL_ARGS "")
  if(BUILD_FOR_HOST)
    set(OPTIONAL_ARGS "-Dprotobuf_WITH_ZLIB=OFF")
  else()
    set(OPTIONAL_ARGS
        "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
        "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
        "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}"
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
        "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
        "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
        "-Dprotobuf_WITH_ZLIB=ON"
        "-DZLIB_ROOT:FILEPATH=${ZLIB_ROOT}"
        ${EXTERNAL_OPTIONAL_ARGS})
    set(OPTIONAL_CACHE_ARGS "-DZLIB_ROOT:STRING=${ZLIB_ROOT}")
  endif()

  set(PROTOBUF_REPOSITORY https://github.com/protocolbuffers/protobuf.git)
  set(PROTOBUF_TAG v3.6.1)

  cache_third_party(
    ${TARGET_NAME}
    REPOSITORY
    ${PROTOBUF_REPOSITORY}
    TAG
    ${PROTOBUF_TAG}
    DIR
    PROTOBUF_SOURCE_DIR)

  ExternalProject_Add(
    ${TARGET_NAME}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    ${SHALLOW_CLONE}
    "${PROTOBUF_DOWNLOAD_CMD}"
    PREFIX ${PROTOBUF_PREFIX_DIR}
    SOURCE_DIR ${PROTOBUF_SOURCE_DIR}
    UPDATE_COMMAND ""
    DEPENDS zlib
    CONFIGURE_COMMAND
      ${CMAKE_COMMAND}
      ${PROTOBUF_SOURCE_DIR}/cmake
      ${OPTIONAL_ARGS}
      -Dprotobuf_BUILD_TESTS=OFF
      -DCMAKE_SKIP_RPATH=ON
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
      -DCMAKE_INSTALL_LIBDIR=lib
      -DBUILD_SHARED_LIBS=OFF
    CMAKE_CACHE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${PROTOBUF_INSTALL_DIR}
      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
      -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      ${OPTIONAL_CACHE_ARGS})
endfunction()

build_protobuf(extern_protobuf FALSE)

message(STATUS "protobuf executable: ${PROTOBUF_PROTOC_EXECUTABLE}")
