// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/print_op.h"
#include "kaleido/core/operators/transpose_op.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tensor_shape.h"

#include <gtest/gtest.h>

#include <iostream>
#include <random>

namespace kaleido {
namespace core {

namespace {
/* load random value in input and compare the value in result with the permuted
 * index */
template <typename T>
void TestTransposeResult(const Tensor& input, const Tensor& result,
                         std::vector<size_t> dims) {
    T* input_cpu = (T*)malloc(input.numel() * sizeof(T));
    T* result_cpu = (T*)malloc(result.numel() * sizeof(T));
    CudaCheck(cudaMemcpy(input_cpu, input.data<T>(), input.numel() * sizeof(T),
                         cudaMemcpyDeviceToHost));
    CudaCheck(cudaMemcpy(result_cpu, result.data<T>(),
                         result.numel() * sizeof(T), cudaMemcpyDeviceToHost));

    std::vector<int64_t> in_stride = {1};
    std::vector<int64_t> out_stride = {1};

    for (int i = dims.size() - 1; i > 0; --i) {
        in_stride.insert(in_stride.begin(), in_stride[0] * input.dim_size(i));
        out_stride.insert(out_stride.begin(),
                          out_stride[0] * result.dim_size(i));
    }

    srand((unsigned)time(NULL));
    for (int round = 0; round < 100000; ++round) {
        int64_t in_idx = 0;
        int64_t out_idx = 0;
        std::vector<int> random_index;
        for (int i = 0; i < dims.size(); ++i) {
            int idx = rand() % input.dim_size(i);
            random_index.push_back(idx);
            in_idx += idx * in_stride[i];
        }
        for (int i = 0; i < dims.size(); ++i) {
            out_idx += out_stride[i] * random_index[dims[i]];
        }
        if (input_cpu[in_idx] != result_cpu[out_idx]) {
            std::cout << "error in: " << std::endl;
            for_each(random_index.cbegin(), random_index.cend(),
                     [](const int& c) { std::cout << c << " "; });
            std::cout << " with input position: " << in_idx
                      << "output value: " << out_idx << std::endl;
            free(input_cpu);
            free(result_cpu);
            return;
        }
    }
    free(input_cpu);
    free(result_cpu);
}

template <typename T>
void TransposeCPU(const Tensor& input, Tensor& output,
                  std::vector<size_t> dims) {
    T* input_cpu = (T*)malloc(input.numel() * sizeof(T));
    T* output_cpu = (T*)malloc(output.numel() * sizeof(T));
    CudaCheck(cudaMemcpy(input_cpu, input.data<T>(), input.numel() * sizeof(T),
                         cudaMemcpyDeviceToHost));

    std::vector<int64_t> in_stride = {1};
    std::vector<int64_t> out_stride = {1};

    for (int i = dims.size() - 1; i > 0; --i) {
        in_stride.insert(in_stride.begin(), in_stride[0] * input.dim_size(i));
        out_stride.insert(out_stride.begin(),
                          out_stride[0] * output.dim_size(i));
    }
    for (int64_t idx = 0; idx < input.numel(); ++idx) {
        int64_t out_idx = 0;
        std::vector<int64_t> in_index;
        int64_t index = idx;
        for (int i = 0; i < dims.size(); ++i) {
            in_index.push_back(index / in_stride[i]);
            index = index % in_stride[i];
        }
        for (int i = 0; i < dims.size(); ++i) {
            out_idx += out_stride[i] * in_index[dims[i]];
        }
        output_cpu[out_idx] = input_cpu[idx];
    }
    CudaCheck(cudaMemcpy(output.mutable_data<T>(), output_cpu,
                         output.numel() * sizeof(T), cudaMemcpyHostToDevice));
}
}  // namespace

TEST(TestTransposeOps, test1) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    Place place = CUDAPlace();

    ops::FillOp<GPUContext, CUDAPlace, float> frand;

    TensorShape shape1{1024};
    TensorTypeDesc desc1 =
        TensorTypeDesc(shape1, PrimitiveType::FLOAT32, place);
    Tensor T1(desc1, allocator);
    frand(T1);

    TensorShape shape2{1024};
    TensorTypeDesc desc2 =
        TensorTypeDesc(shape2, PrimitiveType::FLOAT32, place);
    Tensor T2(desc2, allocator);

    ops::TransposeOp<GPUContext, CUDAPlace, float> transpose;

    std::vector<size_t> dims = {0};
    transpose(T1, T2, dims);
    TestTransposeResult<float>(T1, T2, dims);
}

TEST(TestTransposeOps, test2) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    Place place = CUDAPlace();

    ops::FillOp<GPUContext, CUDAPlace, float> frand;
    ops::PrintOp<GPUContext, CUDAPlace, float> printer;

    TensorShape shape1{2, 4};
    TensorTypeDesc desc1 =
        TensorTypeDesc(shape1, PrimitiveType::FLOAT32, place);
    Tensor T1(desc1, allocator);
    frand(T1);

    TensorShape shape2{4, 2};
    TensorTypeDesc desc2 =
        TensorTypeDesc(shape2, PrimitiveType::FLOAT32, place);
    Tensor T2(desc2, allocator);

    std::cout << printer(T1, T1.numel()) << std::endl;
    std::cout << printer(T2, T2.numel()) << std::endl;

    ops::TransposeOp<GPUContext, CUDAPlace, float> transpose;
    std::vector<size_t> dims = {1, 0};
    transpose(T1, T2, dims);
    TestTransposeResult<float>(T1, T2, dims);
    std::cout << printer(T2, T1.numel()) << std::endl;
}

TEST(TestTransposeOps, test3) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    Place place = CUDAPlace();

    ops::FillOp<GPUContext, CUDAPlace, float> frand;
    ops::PrintOp<GPUContext, CUDAPlace, float> printer;

    TensorShape shape1{2, 4, 8};
    TensorTypeDesc desc1 =
        TensorTypeDesc(shape1, PrimitiveType::FLOAT32, place);
    Tensor T1(desc1, allocator);
    frand(T1);

    TensorShape shape2{4, 8, 2};
    TensorTypeDesc desc2 =
        TensorTypeDesc(shape2, PrimitiveType::FLOAT32, place);
    Tensor T2(desc2, allocator);

    std::cout << printer(T1, T1.numel()) << std::endl;
    std::cout << printer(T2, T2.numel()) << std::endl;

    ops::TransposeOp<GPUContext, CUDAPlace, float> transpose;

    std::vector<size_t> dims = {1, 2, 0};
    transpose(T1, T2, dims);
    TestTransposeResult<float>(T1, T2, dims);
    std::cout << printer(T2, T1.numel()) << std::endl;
}

TEST(TestTransposeOps, test4) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    Place place = CUDAPlace();

    ops::FillOp<GPUContext, CUDAPlace, float> frand;

    TensorShape shape1{256, 512, 2048};
    TensorTypeDesc desc1 =
        TensorTypeDesc(shape1, PrimitiveType::FLOAT32, place);
    Tensor T1(desc1, allocator);
    frand(T1);

    TensorShape shape2{2048, 512, 256};
    TensorTypeDesc desc2 =
        TensorTypeDesc(shape2, PrimitiveType::FLOAT32, place);
    Tensor T2(desc2, allocator);

    ops::TransposeOp<GPUContext, CUDAPlace, float> transpose;

    std::vector<size_t> dims = {2, 1, 0};
    transpose(T1, T2, dims);
    TestTransposeResult<float>(T1, T2, dims);
}

TEST(TestTransposeOps, test5) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    Place place = CUDAPlace();

    ops::FillOp<GPUContext, CUDAPlace, float> frand;

    TensorShape shape1{2, 4, 6, 8, 10};
    TensorTypeDesc desc1 =
        TensorTypeDesc(shape1, PrimitiveType::FLOAT32, place);
    Tensor T1(desc1, allocator);
    frand(T1);

    TensorShape shape2{2, 8, 6, 10, 4};
    TensorTypeDesc desc2 =
        TensorTypeDesc(shape2, PrimitiveType::FLOAT32, place);
    Tensor T2(desc2, allocator);

    ops::TransposeOp<GPUContext, CUDAPlace, float> transpose;

    std::vector<size_t> dims = {0, 3, 2, 4, 1};
    transpose(T1, T2, dims);
    TestTransposeResult<float>(T1, T2, dims);
}

}  // namespace core
}  // namespace kaleido
