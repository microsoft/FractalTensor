include(ExternalProject)

set(CCCL_PREFIX_DIR ${THIRD_PARTY_PATH}/cccl)
set(CCCL_SOURCE_DIR ${CCCL_PREFIX_DIR}/src/extern_cccl)
set(CCCL_REPOSITORY https://github.com/NVIDIA/cccl.git)
set(CCCL_TAG v2.3.0-rc0)

cache_third_party(
  extern_cccl
  REPOSITORY
  ${CCCL_REPOSITORY}
  TAG
  ${CCCL_TAG}
  DIR
  CCCL_SOURCE_DIR)

set(CUB_INCLUDE_DIR ${CCCL_SOURCE_DIR}/cub)
set(THRUST_INCLUDE_DIR ${CCCL_SOURCE_DIR}/thrust)
set(LIBCUDACXX_INCLUDE_DIR ${CCCL_SOURCE_DIR}/libcudacxx)

ExternalProject_Add(
  extern_cccl
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${CCCL_DOWNLOAD_CMD}"
  PREFIX ${CCCL_PREFIX_DIR}
  SOURCE_DIR ${CCCL_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

include_directories(${CUB_INCLUDE_DIR})
include_directories(${THRUST_INCLUDE_DIR})
include_directories(${LIBCUDACXX_INCLUDE_DIR})
