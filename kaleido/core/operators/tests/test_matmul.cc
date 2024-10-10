// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/matmul_op.h"
#include "kaleido/core/operators/tests/test_utils.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tensor_shape.h"
#include "kaleido/core/types.h"

#include <gtest/gtest.h>

namespace kaleido {
namespace core {

using namespace test_utils;

void testMatMul(int rows_A, int cols_A, int rows_B, int cols_B, int rows_C,
                int cols_C, bool transa, bool transb,
                const std::string& layout) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    ops::FillOp<GPUContext, CUDAPlace, float> frand;

    // input1
    Tensor A({rows_A, cols_A}, allocator, "float32", "cuda", layout);
    frand(A);

    // input2
    Tensor B({rows_B, cols_B}, allocator, "float32", "cuda", layout);
    frand(B);

    // output
    Tensor C({rows_C, cols_C}, allocator, "float32", "cuda", layout);
    C.data<float>();

    ops::MatMulOp<GPUContext, CUDAPlace, float> matmul;
    matmul(context, A, transa, B, transb, C);

    float* dA = (float*)malloc(A.numel() * sizeof(float));
    float* dB = (float*)malloc(B.numel() * sizeof(float));
    float* dC = (float*)malloc(C.numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dA, A.data<float>(), A.numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CudaCheck(cudaMemcpy(dB, B.data<float>(), B.numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CudaCheck(cudaMemcpy(dC, C.data<float>(), C.numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    int m = transa ? A.dim_size(1) : A.dim_size(0);
    int n = transb ? B.dim_size(0) : B.dim_size(1);
    int k = transa ? A.dim_size(0) : A.dim_size(1);

    if (layout == "col")
        CompareWithNaiveMatMulColMajor(dA, transa, dB, transb, dC, m, n, k);
    else
        CompareWithNaiveMatMulRowMajor(dA, transa, dB, transb, dC, m, n, k);

    free(dA);
    free(dB);
    free(dC);
}

TEST(TestMatMul, testColMajor) {
    std::string layout = "col";
    // NN
    testMatMul(16, 34, 34, 17, 16, 17, false, false, layout);
    // NT
    testMatMul(16, 34, 17, 34, 16, 17, false, true, layout);
    // TN
    testMatMul(7, 11, 7, 9, 11, 9, true, false, layout);
    // TT
    testMatMul(11, 7, 13, 11, 7, 13, true, true, layout);
}

TEST(TestMatMul, testRowMajor) {
    std::string layout = "row";
    // NN
    testMatMul(16, 34, 34, 17, 16, 17, false, false, layout);
    // NT
    testMatMul(16, 34, 17, 34, 16, 17, false, true, layout);
    // TN
    testMatMul(7, 11, 7, 9, 11, 9, true, false, layout);
    // TT
    testMatMul(11, 7, 13, 11, 7, 13, true, true, layout);
}

}  // namespace core
}  // namespace kaleido
