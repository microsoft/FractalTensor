// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/fractal_tensor.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/print_op.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tensor_shape.h"

#include <gtest/gtest.h>

#include <iostream>

namespace kaleido {
namespace core {

TEST(TestTypeDesc, test1) {
  TensorShape shape1{3, 4, 5};
  EXPECT_EQ(shape1.ndim(), 3);
  EXPECT_EQ(shape1.numel(), 60);

  TensorShape shape2{3, 4, 5};
  EXPECT_TRUE(shape1 == shape2);

  TensorShape shape3{3, 4};
  EXPECT_FALSE(shape1 == shape3);
  EXPECT_TRUE(shape1 != shape3);
}

TEST(TestTensorCreation, test1) {
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  auto allocator = std::make_shared<CudaMemoryPool>();
  allocator->add_track_stream(stream);

  Place place = CUDAPlace();

  TensorShape shape{64, 128};
  TensorTypeDesc desc = TensorTypeDesc(shape, PrimitiveType::FLOAT32, place);

  Tensor t(desc, allocator);
  t.data<float>();
  std::cout << t << std::endl;

  ops::PrintOp<GPUContext, CUDAPlace, float> printer;
  ops::FillOp<GPUContext, CUDAPlace, float> f;
  f(t, 5.);
  std::cout << printer(t, 10) << std::endl;
  ops::FillOp<GPUContext, CUDAPlace, float> f2;
  f2(t);
  std::cout << printer(t, 15) << std::endl;
}

TEST(TestFractalTensorCreation, test1) {
  Place place = CUDAPlace();
  TensorShape shape{64, 64};
  TensorTypeDesc tensor_desc(shape, PrimitiveType::FLOAT32, place);

  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));
  auto allocator = std::make_shared<CudaMemoryPool>();
  allocator->add_track_stream(stream);

  // Type description for a depth-1 FractalTensor with uninitialized indexing
  // tree.
  FractalTensorTypeDesc ft_desc1(tensor_desc);
  FractalTensor ft1(ft_desc1, allocator);
  std::cout << ft1 << std::endl;

  // Type description for a depth-2 FractalTensor with initialized indexing
  // tree.
  FractalTensorTypeDesc ft_desc2(tensor_desc, {{0, 7, 11}}, {false});
  FractalTensor ft2(ft_desc2, allocator);
  std::cout << ft2 << std::endl;

  // Type description for a depth-3 FractalTensor with initialized indexing
  // tree.
  FractalTensorTypeDesc ft_desc3(
      tensor_desc,
      {{0, 3}, {0, 4, 7, 9}, {0, 1, 5, 14, 19, 23, 30, 33, 43, 50}},
      {false, false, false});
  FractalTensor ft3(ft_desc3, allocator);
  std::cout << ft3 << std::endl;
}

TEST(TestReshape, test1) {
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  auto allocator = std::make_shared<CudaMemoryPool>();
  allocator->add_track_stream(stream);

  ops::FillOp<GPUContext, CUDAPlace, float> rand;

  Tensor t1({16, 17, 32}, allocator);
  t1.data<float>();
  rand(t1);
  std::cout << t1 << std::endl;

  auto t2 = Tensor::ReshapeFrom<float>(t1, {16 * 17, 32});
}

}  // namespace core
}  // namespace kaleido
