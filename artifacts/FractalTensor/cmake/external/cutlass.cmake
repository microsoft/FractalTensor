include(ExternalProject)

set(CUTLASS_PREFIX_DIR ${THIRD_PARTY_PATH}/cutlass)
set(CUTLASS_SOURCE_DIR ${CUTLASS_PREFIX_DIR}/src/extern_cutlass)
set(CUTLASS_REPOSITORY https://github.com/NVIDIA/cutlass.git)
set(CUTLASS_TAG v3.2.2)

cache_third_party(
  extern_cutlass
  REPOSITORY
  ${CUTLASS_REPOSITORY}
  TAG
  ${CUTLASS_TAG}
  DIR
  CUTLASS_SOURCE_DIR)

set(CUTLASS_INCLUDE_DIR "${CUTLASS_SOURCE_DIR}/include")
include_directories(${CUTLASS_INCLUDE_DIR})
include_directories("${CUTLASS_SOURCE_DIR}/tools/util/include")

ExternalProject_Add(
  extern_cutlass
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${CUTLASS_DOWNLOAD_CMD}"
  PREFIX ${CUTLASS_PREFIX_DIR}
  SOURCE_DIR ${CUTLASS_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  TEST_COMMAND "")

add_library(cutlass INTERFACE)
target_include_directories(cutlass INTERFACE ${ROOT_DIR}/include
                                             ${ROOT_DIR}/tools/util/include)
target_link_libraries(cutlass INTERFACE CUDA::cudart)
target_include_directories(cutlass
                           INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_dependencies(cutlass extern_cutlass)
