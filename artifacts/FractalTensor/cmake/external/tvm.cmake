include(ExternalProject)

set(TVM_PREFIX_DIR ${THIRD_PARTY_PATH}/tvm)
set(TVM_SOURCE_DIR ${TVM_PREFIX_DIR}/src/extern_tvm)

set(TVM_REPOSITORY https://github.com/apache/tvm.git)
set(TVM_TAG v0.8.0)

cache_third_party(
  extern_tvm
  REPOSITORY
  ${TVM_REPOSITORY}
  TAG
  ${TVM_TAG}
  DIR
  TVM_SOURCE_DIR)

set(TVM_INCLUDE_DIR ${TVM_SOURCE_DIR}/include)
include_directories(${TVM_INCLUDE_DIR})

ExternalProject_Add(
  extern_tvm
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${TVM_DOWNLOAD_CMD}"
  PREFIX ${TVM_PREFIX_DIR}
  SOURCE_DIR ${TVM_SOURCE_DIR}
  BUILD_IN_SOURCE 1
  COMMAND "cp ${TVM_SOURCE_DIR}/cmake/config.cmake ${TVM_SOURCE_DIR}"
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -DCMAKE_POSITION_INDEPENDENT_CODE=ON .
  BUILD_COMMAND $(MAKE) -j$(nproc)
  INSTALL_COMMAND ""
  TEST_COMMAND "")
