# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

include(ExternalProject)

set(GFLAGS_PREFIX_DIR ${THIRD_PARTY_PATH}/gflags)
set(GFLAGS_SOURCE_DIR ${THIRD_PARTY_PATH}/gflags/src/extern_gflags)
set(GFLAGS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gflags)
set(GFLAGS_INCLUDE_DIR
    "${GFLAGS_INSTALL_DIR}/include"
    CACHE PATH "gflags include directory." FORCE)
set(GFLAGS_REPOSITORY https://github.com/gflags/gflags.git)
set(GFLAGS_TAG 77592648e3f3be87d6c7123eb81cbad75f9aef5a)
set(GFLAGS_LIBRARIES
    "${GFLAGS_INSTALL_DIR}/lib/libgflags.a"
    CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
set(BUILD_COMMAND $(MAKE) --silent)
set(INSTALL_COMMAND $(MAKE) install)

include_directories(${GFLAGS_INCLUDE_DIR})

cache_third_party(
  extern_gflags
  REPOSITORY
  ${GFLAGS_REPOSITORY}
  TAG
  ${GFLAGS_TAG}
  DIR
  GFLAGS_SOURCE_DIR)

ExternalProject_Add(
  extern_gflags
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${GFLAGS_DOWNLOAD_CMD}"
  PREFIX ${GFLAGS_PREFIX_DIR}
  SOURCE_DIR ${GFLAGS_SOURCE_DIR}
  BUILD_COMMAND ${BUILD_COMMAND}
  INSTALL_COMMAND ${INSTALL_COMMAND}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
             -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
             -DBUILD_STATIC_LIBS=ON
             -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DBUILD_TESTING=OFF
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE})

add_library(gflags STATIC IMPORTED GLOBAL)
set_property(TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIBRARIES})
add_dependencies(gflags extern_gflags)
