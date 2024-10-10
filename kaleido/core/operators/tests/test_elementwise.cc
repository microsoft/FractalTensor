// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/math_functor.h"
#include "kaleido/core/operators/elementwise_op.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tensor_shape.h"

#include <gtest/gtest.h>

namespace kaleido {
namespace core {

template <typename Functor>
void TestBinaryElementwise(TensorShape& shape, Functor func) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    ops::FillOp<GPUContext, CUDAPlace, float> frand;

    // input1
    TensorTypeDesc desc1 = TensorTypeDesc(shape, PrimitiveType::FLOAT32, place);
    Tensor x1(desc1, allocator);
    frand(x1);

    // input2
    TensorTypeDesc desc2 = TensorTypeDesc(shape, PrimitiveType::FLOAT32, place);
    Tensor x2(desc2, allocator);
    frand(x2);

    std::vector<Tensor> inputs;
    inputs.emplace_back(x1);
    inputs.emplace_back(x2);

    // output
    TensorTypeDesc desc = TensorTypeDesc(shape, PrimitiveType::FLOAT32, place);
    Tensor z(desc, allocator);
    z.data<float>();

    ops::ElementwiseOp<GPUContext, CUDAPlace, ops::ElementwiseType::kBinary,
                       float, Functor>
        f;
    f(context, inputs, z, func);

    float* dx1 = (float*)malloc(x1.numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dx1, x1.data<float>(), x1.numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float* dx2 = (float*)malloc(x2.numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dx2, x2.data<float>(), x2.numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float* dz = (float*)malloc(z.numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dz, z.data<float>(), z.numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < z.numel(); ++i)
        EXPECT_FLOAT_EQ(dz[i], func(dx1[i], dx2[i]));

    free(dz);
}

template <typename Functor>
void TestUnaryElementwise(TensorShape& shape, Functor func) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    ops::FillOp<GPUContext, CUDAPlace, float> frand;

    TensorTypeDesc desc1 = TensorTypeDesc(shape, PrimitiveType::FLOAT32, place);
    Tensor x(desc1, allocator);
    frand(x);

    std::vector<Tensor> inputs;
    inputs.emplace_back(x);

    TensorTypeDesc desc = TensorTypeDesc(shape, PrimitiveType::FLOAT32, place);
    Tensor z(desc, allocator);
    z.data<float>();

    ops::ElementwiseOp<GPUContext, CUDAPlace, ops::ElementwiseType::kUnary,
                       float, Functor>
        f;
    f(context, inputs, z, func);

    float* dx = (float*)malloc(x.numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dx, x.data<float>(), x.numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float* dz = (float*)malloc(z.numel() * sizeof(float));
    CudaCheck(cudaMemcpy(dz, z.data<float>(), z.numel() * sizeof(float),
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < z.numel(); ++i) EXPECT_FLOAT_EQ(dz[i], func(dx[i]));

    free(dz);
}

TEST(TestElementwise, add) {
    cuda_kernel::Add<float> add;
    TensorShape shape1{3, 7, 5};
    TensorShape shape2{233, 11};
    TensorShape shape3{1027, 33, 11, 5};

    TestBinaryElementwise<cuda_kernel::Add<float>>(shape1, add);
    TestBinaryElementwise<cuda_kernel::Add<float>>(shape2, add);
    TestBinaryElementwise<cuda_kernel::Add<float>>(shape3, add);

    cuda_kernel::Exp<float> exp;
    TestUnaryElementwise<cuda_kernel::Exp<float>>(shape1, exp);
    TestUnaryElementwise<cuda_kernel::Exp<float>>(shape2, exp);
    TestUnaryElementwise<cuda_kernel::Exp<float>>(shape3, exp);
}

}  // namespace core
}  // namespace kaleido
