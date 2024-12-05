set(TEST_ROOT ${PROJECT_SOURCE_DIR}/kaleido/core/operators/tests)

cc_test_build(
  test_concat
  SRCS
  ${TEST_ROOT}/test_concat.cc
  DEPS
  concat_op
  print_op
  fill_op
  fractaltensor_core)

cc_test_build(
  test_matmul
  SRCS
  ${TEST_ROOT}/test_matmul.cc
  DEPS
  matmul_op
  fill_op
  fractaltensor_core)

cc_test_build(
  test_softmax
  SRCS
  ${TEST_ROOT}/test_softmax.cc
  DEPS
  softmax_op
  online_softmax_op
  expect_eq_op
  fractaltensor_core
  fill_op)

cc_test_build(
  test_gemm_batched
  SRCS
  ${TEST_ROOT}/test_gemm_batched.cc
  DEPS
  gemm_batched_op
  fill_op
  fractaltensor_core)

cc_test_build(
  test_transpose
  SRCS
  ${TEST_ROOT}/test_transpose.cc
  DEPS
  transpose_op
  print_op
  fill_op
  fractaltensor_core)

cc_test_build(
  test_elementwise
  SRCS
  ${TEST_ROOT}/test_elementwise.cc
  DEPS
  elementwise_op
  print_op
  fill_op
  fractaltensor_core)

cc_test_build(
  test_gather_scatter
  SRCS
  ${TEST_ROOT}/test_gather_scatter.cc
  DEPS
  gather_nd_op
  scatter_nd_op
  fill_op
  print_op
  fractaltensor_core)

nv_test(
  test_mm_tile_access
  SRCS
  ${TEST_ROOT}/test_mm_tile_access.cu
  DEPS
  fill_op
  print_op
  expect_eq_op
  fractaltensor_core)

nv_test(
  test_cutlass_mm
  SRCS
  ${TEST_ROOT}/test_cutlass_gemm.cu
  DEPS
  fill_op
  print_op
  expect_eq_op
  fractaltensor_core)

nv_test(
  test_b2b_gemm
  SRCS
  ${TEST_ROOT}/test_b2b_gemm.cu
  DEPS
  fill_op
  print_op
  expect_eq_op
  fractaltensor_core)

nv_test(
  test_lstm
  SRCS
  ${TEST_ROOT}/test_lstm.cu
  DEPS
  fill_op
  print_op
  expect_eq_op
  fractaltensor_core)

nv_test(
  test_batched_lstm
  SRCS
  ${TEST_ROOT}/test_batched_lstm.cu
  DEPS
  fill_op
  print_op
  expect_eq_op
  fractaltensor_core)

nv_test(
  test_scatter_nd
  SRCS
  ${TEST_ROOT}/test_scatter_nd.cu
  DEPS
  fill_op
  print_op
  expect_eq_op
  fractaltensor_core)

nv_test(
  test_stacked_lstm_region
  SRCS
  ${TEST_ROOT}/test_stacked_lstm_region.cu
  DEPS
  fill_op
  print_op
  expect_eq_op
  fractaltensor_core)

nv_test(
  test_dilated_lstm_region
  SRCS
  ${TEST_ROOT}/test_dilated_lstm_region.cu
  DEPS
  fill_op
  print_op
  expect_eq_op
  fractaltensor_core)
