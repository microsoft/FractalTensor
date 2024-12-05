set(TARGET fractaltensor_core)

include_directories("${PROJECT_SOURCE_DIR}")
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUDNN_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# set(PATH_PREFIX ${PROJECT_SOURCE_DIR}/kaleido/core) file(GLOB_RECURSE
# PROTOBUF_FILE "${PATH_PREFIX}/*.proto") get_filename_component(PROTO_PATH
# ${PROTOBUF_FILE} ABSOLUTE) get_filename_component(PROTO_NAME ${PROTOBUF_FILE}
# NAME_WE)

# cpp_proto_generate("${TARGET}_proto" SRCS "${PROTOBUF_FILE}")
# add_custom_command( TARGET "${TARGET}_proto" POST_BUILD COMMAND cp *.pb.*
# ${PATH_PREFIX} COMMENT "Copy generated C++ proto into directory kaleido/core."
# WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
#
# py_proto_generate("${TARGET}_proto_py" SRCS "${PROTOBUF_FILE}")
# add_custom_command( TARGET "${TARGET}_proto_py" POST_BUILD COMMAND cp *.py
# "${PROJECT_SOURCE_DIR}/kaleido/frontend" COMMENT "Copy generated python proto
# into directory kaleido/frontend." WORKING_DIRECTORY
# ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(operators)

file(
  GLOB KALEIDO_CORE_SRCS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc" "device/*.cc")

cc_library(
  ${TARGET}
  SHARED
  IMPORTED
  SRCS
  ${KALEIDO_CORE_SRCS}
  DEPS
  python
  # ${TARGET}_proto protobuf
)

target_link_libraries(${TARGET} Boost::filesystem Boost::regex)
target_link_libraries(${TARGET} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}
                      ${CUDNN_LIBRARIES})

add_subdirectory(tests)
add_subdirectory(device/tests)
