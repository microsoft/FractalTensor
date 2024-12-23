# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

include(ExternalProject)

set(GLOG_PREFIX_DIR ${THIRD_PARTY_PATH}/glog)
set(GLOG_SOURCE_DIR ${THIRD_PARTY_PATH}/glog/src/extern_glog)
set(GLOG_INSTALL_DIR ${THIRD_PARTY_PATH}/install/glog)
set(GLOG_INCLUDE_DIR
    "${GLOG_INSTALL_DIR}/include"
    CACHE PATH "glog include directory." FORCE)
set(GLOG_REPOSITORY https://github.com/google/glog.git)
set(GLOG_TAG v0.3.5)

set(GLOG_LIBRARIES
    "${GLOG_INSTALL_DIR}/lib/libglog.a"
    CACHE FILEPATH "glog library." FORCE)
set(GLOG_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

include_directories(${GLOG_INCLUDE_DIR})

cache_third_party(
  extern_glog
  REPOSITORY
  ${GLOG_REPOSITORY}
  TAG
  ${GLOG_TAG}
  DIR
  GLOG_SOURCE_DIR)

ExternalProject_Add(
  extern_glog
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${GLOG_DOWNLOAD_CMD}"
  DEPENDS gflags
  PREFIX ${GLOG_PREFIX_DIR}
  SOURCE_DIR ${GLOG_SOURCE_DIR}
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_CXX_FLAGS=${GLOG_CMAKE_CXX_FLAGS}
             -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
             -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
             -DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}
             -DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}
             -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR}
             -DCMAKE_INSTALL_LIBDIR=${GLOG_INSTALL_DIR}/lib
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON
             -DWITH_GFLAGS=ON
             -Dgflags_DIR=${GFLAGS_INSTALL_DIR}/lib/cmake/gflags
             -DBUILD_TESTING=OFF
             -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
             ${EXTERNAL_OPTIONAL_ARGS}
  CMAKE_CACHE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${GLOG_INSTALL_DIR}
    -DCMAKE_INSTALL_LIBDIR:PATH=${GLOG_INSTALL_DIR}/lib
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE})

add_library(glog SHARED IMPORTED GLOBAL)
set_property(TARGET glog PROPERTY IMPORTED_LOCATION ${GLOG_LIBRARIES})
add_dependencies(glog extern_glog gflags)
link_libraries(glog gflags)
