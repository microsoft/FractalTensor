// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/expect_eq_op.h"
#include "kaleido/core/operators/online_softmax_op.h"
#include "kaleido/core/operators/softmax_op.h"
#include "kaleido/core/operators/tests/test_utils.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"

#include <gtest/gtest.h>

namespace kaleido {
namespace core {

using namespace test_utils;

void TestSoftmax(int rows, int cols) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    std::shared_ptr<Tensor> x = RandomGpuTensor<float>({rows, cols}, allocator);
    std::shared_ptr<Tensor> y =
        ConstantGpuTensor<float>({rows, cols}, allocator, 0.);

    ops::SoftmaxOp<GPUContext, CUDAPlace, float> softmax;
    softmax(context, *x, *y, 0);

    float* dy = (float*)malloc(y->numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dy, y->data<float>(), y->numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < y->dim_size(0); ++i) {
        float tmp = 0.;
        for (int j = 0; j < y->dim_size(1); ++j)
            tmp += dy[i * y->dim_size(1) + j];

        EXPECT_NEAR(tmp, 1, 1e-3);
    }
    free(dy);
}

void TestOnlineSoftmax(int rows, int cols) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    std::shared_ptr<Tensor> x = RandomGpuTensor<float>({rows, cols}, allocator);
    std::shared_ptr<Tensor> y1 =
        ConstantGpuTensor<float>({rows, cols}, allocator, 0.);

    std::shared_ptr<Tensor> y2 =
        ConstantGpuTensor<float>({rows, cols}, allocator, 0.);

    ops::SoftmaxOp<GPUContext, CUDAPlace, float> softmax;
    softmax(context, *x, *y1, 0);

    ops::OnlineNormalizedSoftmaxOp<GPUContext, CUDAPlace, float> online_softmax;
    online_softmax(context, *x, *y2, 0);

    // FIXME(Ying): the numeric difference is too large.
    ops::ExpectEqOp<GPUContext, CUDAPlace, float> check;
    check(*y1, *y2, 1e-1);
}

TEST(TestSoftmaxOps, softmax) {
    TestSoftmax(2, 7);
    TestSoftmax(2, 517);
    TestSoftmax(2, 1027);
}

TEST(TestOnlineSoftmax, online_softmax) {
    TestOnlineSoftmax(2, 7);
    TestOnlineSoftmax(2, 517);
    TestOnlineSoftmax(2, 1027);
}

}  // namespace core
}  // namespace kaleido
