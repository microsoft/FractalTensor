// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/gemm_batched_op.h"
#include "kaleido/core/operators/tests/test_utils.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tensor_shape.h"
#include "kaleido/core/types.h"

#include <gtest/gtest.h>

namespace kaleido {
namespace core {

using namespace test_utils;

void testGemmBatched(int rows_A, int cols_A, int rows_B, int cols_B, int rows_C,
                     int cols_C, bool transa, bool transb, std::string& layout,
                     int batch_count) {
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  auto allocator = std::make_shared<CudaMemoryPool>();
  allocator->add_track_stream(stream);

  CUDAPlace place = CUDAPlace(0);
  GPUContext context(place);

  ops::FillOp<GPUContext, CUDAPlace, float> frand;

  std::vector<Tensor> a_array;
  std::vector<Tensor> b_array;
  std::vector<Tensor> c_array;

  for (int i = 0; i < batch_count; ++i) {
    // input1
    Tensor* A =
        new Tensor({rows_A, cols_A}, allocator, "float32", "cuda", layout);
    frand(*A);
    a_array.emplace_back(*A);

    // input2
    Tensor* B =
        new Tensor({rows_B, cols_B}, allocator, "float32", "cuda", layout);
    frand(*B);
    b_array.emplace_back(*B);

    // output
    Tensor* C =
        new Tensor({rows_C, cols_C}, allocator, "float32", "cuda", layout);
    C->data<float>();
    c_array.emplace_back(*C);
  }

  ops::GemmBatchedOp<GPUContext, CUDAPlace, float> gemm_batched;
  gemm_batched(context, a_array, transa, b_array, transb, c_array);

  for (int i = 0; i < batch_count; ++i) {
    float* dA = (float*)malloc(a_array[i].numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dA, a_array[i].data<float>(),
                         a_array[i].numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float* dB = (float*)malloc(b_array[i].numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dB, b_array[i].data<float>(),
                         b_array[i].numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float* dC = (float*)malloc(c_array[i].numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dC, c_array[i].data<float>(),
                         c_array[i].numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    int m = transa ? a_array[0].dim_size(1) : a_array[0].dim_size(0);
    int n = transb ? b_array[0].dim_size(0) : b_array[0].dim_size(1);
    int k = transa ? a_array[0].dim_size(0) : a_array[0].dim_size(1);
    if (layout == "col")
      CompareWithNaiveMatMulColMajor(dA, transa, dB, transb, dC, m, n, k);
    else
      CompareWithNaiveMatMulRowMajor(dA, transa, dB, transb, dC, m, n, k);

    free(dA);
    free(dB);
    free(dC);
  }
}

TEST(TestGemmBatched, testColMajor) {
  std::string layout = "col";
  // NN
  testGemmBatched(16, 34, 34, 17, 16, 17, false, false, layout, 4);
  // NT
  testGemmBatched(16, 34, 17, 34, 16, 17, false, true, layout, 7);
  // TN
  testGemmBatched(7, 11, 7, 9, 11, 9, true, false, layout, 5);
  // TT
  testGemmBatched(11, 7, 13, 11, 7, 13, true, true, layout, 11);
}

TEST(TestGemmBatched, testRowMajor) {
  std::string layout = "row";
  // NN
  testGemmBatched(16, 34, 34, 17, 16, 17, false, false, layout, 11);
  // NT
  testGemmBatched(16, 34, 17, 34, 16, 17, false, true, layout, 3);
  // TN
  testGemmBatched(7, 11, 7, 9, 11, 9, true, false, layout, 2);
  // TT
  testGemmBatched(11, 7, 13, 11, 7, 13, true, true, layout, 1);
}

}  // namespace core
}  // namespace kaleido
