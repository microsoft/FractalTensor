// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/concat_op.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/print_op.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tensor_shape.h"

#include <gtest/gtest.h>

#include <iostream>

namespace kaleido {
namespace core {

TEST(TestConcatOps, test1) {
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  auto allocator = std::make_shared<CudaMemoryPool>();
  allocator->add_track_stream(stream);

  CUDAPlace place = CUDAPlace(0);
  GPUContext context(place);

  ops::FillOp<GPUContext, CUDAPlace, float> frand;
  ops::PrintOp<GPUContext, CUDAPlace, float> printer;

  // input1
  TensorShape shape1{2, 5, 7};
  TensorTypeDesc desc1 = TensorTypeDesc(shape1, PrimitiveType::FLOAT32, place);
  Tensor t1(desc1, allocator);
  frand(t1);

  std::vector<Tensor> tensors;
  tensors.emplace_back(t1);

  for (auto t : tensors) {
    std::cout << printer(t, t.numel()) << std::endl;
  }

  // output
  TensorShape outShape{2, 5, 7};
  TensorTypeDesc desc = TensorTypeDesc(outShape, PrimitiveType::FLOAT32, place);
  Tensor out(desc, allocator);
  out.data<float>();

  ops::ConcatOp<GPUContext, CUDAPlace, float> cat;
  cat(context, tensors, out, 0);

  std::cout << printer(out, out.numel()) << std::endl;
}

TEST(TestConcatOps, test2) {
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  auto allocator = std::make_shared<CudaMemoryPool>();
  allocator->add_track_stream(stream);

  CUDAPlace place = CUDAPlace(0);
  GPUContext context(place);

  ops::FillOp<GPUContext, CUDAPlace, float> frand;
  ops::PrintOp<GPUContext, CUDAPlace, float> printer;

  // input1
  TensorShape shape1{2, 5};
  TensorTypeDesc desc1 = TensorTypeDesc(shape1, PrimitiveType::FLOAT32, place);
  Tensor t1(desc1, allocator);
  frand(t1);

  // input2
  TensorShape shape2{3, 5};
  TensorTypeDesc desc2 = TensorTypeDesc(shape2, PrimitiveType::FLOAT32, place);
  Tensor t2(desc2, allocator);
  frand(t2);

  std::vector<Tensor> tensors;
  tensors.emplace_back(t1);
  tensors.emplace_back(t2);

  for (auto t : tensors) {
    std::cout << printer(t, t.numel()) << std::endl;
  }

  // output
  TensorShape outShape{5, 5};
  TensorTypeDesc desc = TensorTypeDesc(outShape, PrimitiveType::FLOAT32, place);
  Tensor out(desc, allocator);
  out.data<float>();

  ops::ConcatOp<GPUContext, CUDAPlace, float> cat;
  cat(context, tensors, out, 0);

  std::cout << printer(out, 25) << std::endl;
}

TEST(TestConcatOps, test3) {
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  auto allocator = std::make_shared<CudaMemoryPool>();
  allocator->add_track_stream(stream);

  CUDAPlace place = CUDAPlace(0);
  GPUContext context(place);

  ops::FillOp<GPUContext, CUDAPlace, float> frand;
  ops::PrintOp<GPUContext, CUDAPlace, float> printer;

  // input1
  TensorShape shape1{1, 4};
  TensorTypeDesc desc1 = TensorTypeDesc(shape1, PrimitiveType::FLOAT32, place);
  Tensor t1(desc1, allocator);
  t1.data<float>();
  frand(t1);

  // input2
  TensorShape shape2{1, 7};
  TensorTypeDesc desc2 = TensorTypeDesc(shape2, PrimitiveType::FLOAT32, place);
  Tensor t2(desc2, allocator);
  t2.data<float>();
  frand(t2);

  // input3
  TensorShape shape3{1, 9};
  TensorTypeDesc desc3 = TensorTypeDesc(shape3, PrimitiveType::FLOAT32, place);
  Tensor t3(desc3, allocator);
  t3.data<float>();
  frand(t3);

  std::vector<Tensor> tensors;
  tensors.emplace_back(t1);
  tensors.emplace_back(t2);
  tensors.emplace_back(t3);

  for (auto t : tensors) {
    std::cout << printer(t, t.numel()) << std::endl;
  }

  // output
  TensorShape outShape{1, 20};
  TensorTypeDesc desc = TensorTypeDesc(outShape, PrimitiveType::FLOAT32, place);
  Tensor out(desc, allocator);
  out.data<float>();
  ops::ConcatOp<GPUContext, CUDAPlace, float> cat;
  cat(context, tensors, out, 1);

  std::cout << printer(out, out.numel()) << std::endl;
}

}  // namespace core
}  // namespace kaleido
