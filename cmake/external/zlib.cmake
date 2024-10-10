# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

include(ExternalProject)

set(ZLIB_PREFIX_DIR ${THIRD_PARTY_PATH}/zlib)
set(ZLIB_SOURCE_DIR ${THIRD_PARTY_PATH}/zlib/src/extern_zlib)
set(ZLIB_INSTALL_DIR ${THIRD_PARTY_PATH}/install/zlib)
set(ZLIB_ROOT
    ${ZLIB_INSTALL_DIR}
    CACHE FILEPATH "zlib root directory." FORCE)
set(ZLIB_INCLUDE_DIR
    "${ZLIB_INSTALL_DIR}/include"
    CACHE PATH "zlib include directory." FORCE)
set(ZLIB_REPOSITORY https://github.com/madler/zlib.git)
set(ZLIB_TAG v1.2.8)

include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${THIRD_PARTY_PATH}/install)

cache_third_party(
  extern_zlib
  REPOSITORY
  ${ZLIB_REPOSITORY}
  TAG
  ${ZLIB_TAG}
  DIR
  ZLIB_SOURCE_DIR)

ExternalProject_Add(
  extern_zlib
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${ZLIB_DOWNLOAD_CMD}"
  PREFIX ${ZLIB_PREFIX_DIR}
  SOURCE_DIR ${ZLIB_SOURCE_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR}
             -DBUILD_SHARED_LIBS=OFF
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DCMAKE_MACOSX_RPATH=ON
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${ZLIB_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE})
set(ZLIB_LIBRARIES
    "${ZLIB_INSTALL_DIR}/lib/libz.a"
    CACHE FILEPATH "zlib library." FORCE)

add_library(zlib STATIC IMPORTED GLOBAL)
set_property(TARGET zlib PROPERTY IMPORTED_LOCATION ${ZLIB_LIBRARIES})
add_dependencies(zlib extern_zlib)
