# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

op_library(fill_op DEPS ${CUDA_curand_LIBRARY})
op_library(concat_op)
op_library(transpose_op)
op_library(print_op)
op_library(matmul_op)
op_library(softmax_op)
op_library(gemm_batched_op)
op_library(elementwise_op)
op_library(gather_nd_op)
op_library(scatter_nd_op)
op_library(online_softmax_op)
op_library(expect_eq_op)
add_subdirectory(tests)
