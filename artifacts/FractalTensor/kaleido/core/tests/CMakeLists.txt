set(TEST_ROOT ${PROJECT_SOURCE_DIR}/kaleido/core/tests)

cc_test_build(test_allocator SRCS ${TEST_ROOT}/test_allocator.cc DEPS
              fractaltensor_core)
cc_test_build(
  test_tensor_and_fractaltensor
  SRCS
  ${TEST_ROOT}/test_tensor_and_fractaltensor.cc
  DEPS
  fractaltensor_core
  print_op
  fill_op)

cc_test_build(test_cuda_info SRCS ${TEST_ROOT}/test_cuda_info.cc DEPS
              fractaltensor_core)

cc_test_build(test_layout SRCS ${TEST_ROOT}/test_layout.cc DEPS
              fractaltensor_core)
