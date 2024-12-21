// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/gather_nd_op.h"
#include "kaleido/core/operators/print_op.h"
#include "kaleido/core/operators/scatter_nd_op.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tensor_shape.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <experimental/random>
#include <iostream>
#include <random>

namespace kaleido {
namespace core {

std::vector<int64_t> computeOutputShape(
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& input_dims) {
  size_t index_dims_size = index_dims.size();
  size_t input_dims_size = input_dims.size();
  int64_t end_size = index_dims[index_dims_size - 1];
  auto out_shape =
      std::vector<int64_t>(index_dims.begin(), index_dims.end() - 1);
  for (size_t i = 0; i < input_dims_size - end_size; ++i)
    out_shape.push_back(input_dims[index_dims_size + i]);
  return out_shape;
}

template <typename T>
void checkGatherNdResults(const std::vector<int64_t>& data_shape,
                          const int64_t* ids,
                          const std::vector<int64_t>& indices_shape,
                          const T* cpu_output, size_t slice_size) {
  int64_t end_size = indices_shape.back();
  int64_t numel = std::accumulate(indices_shape.begin(), indices_shape.end(), 1,
                                  std::multiplies<int64_t>());

  size_t ndim = data_shape.size();
  std::vector<int64_t> strides(ndim, 1);
  for (size_t i = ndim - 1; i > 0; --i)
    strides[i - 1] = strides[i] * data_shape[i];

  int count = numel / end_size;
  for (size_t i = 0; i < count; ++i) {
    int start_ids = 0;
    for (int k = 0; k < end_size - 1; ++k)
      start_ids += ids[i * end_size + k] * strides[k];
    start_ids += ids[(i + 1) * end_size - 1] * strides[end_size - 1];

    for (size_t j = 0; j < slice_size; ++j)
      EXPECT_EQ(start_ids + j, cpu_output[i * slice_size + j]);
  }
}

template <typename T>
void checkScatterNdResults(const T* data,
                           const std::vector<int64_t>& data_shape,
                           const int64_t* ids,
                           const std::vector<int64_t>& indices_shape,
                           const T* updates, size_t slice_size) {
  int64_t end_size = indices_shape.back();
  int64_t numel = std::accumulate(indices_shape.begin(), indices_shape.end(), 1,
                                  std::multiplies<int64_t>());
  size_t ndim = data_shape.size();
  std::vector<int64_t> strides(ndim, 1);
  for (size_t i = ndim - 1; i > 0; --i)
    strides[i - 1] = strides[i] * data_shape[i];

  int count = numel / end_size;
  for (size_t i = 0; i < count; ++i) {
    int start_ids = 0;
    for (int k = 0; k < end_size - 1; ++k)
      start_ids += ids[i * end_size + k] * strides[k];
    start_ids += ids[(i + 1) * end_size - 1] * strides[end_size - 1];

    for (size_t j = 0; j < slice_size; ++j)
      EXPECT_FLOAT_EQ(data[start_ids + j], updates[i * slice_size + j] + 5.);
  }
}

TEST(TestGatherNd, test) {
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  auto allocator = std::make_shared<CudaMemoryPool>();
  allocator->add_track_stream(stream);

  CUDAPlace place = CUDAPlace(0);
  GPUContext context(place);

  TensorShape data_shape{1027, 167, 239};  // rank(data) = 3
  TensorTypeDesc data_desc =
      TensorTypeDesc(data_shape, PrimitiveType::INT64, place);
  Tensor data(data_desc, allocator);

  int* cpu_data = (int*)malloc(data.numel() * sizeof(int));
  for (int i = 0; i < data.numel(); ++i) cpu_data[i] = i;
  CudaCheck(cudaMemcpy(data.mutable_data<int>(), cpu_data,
                       data.numel() * sizeof(int), cudaMemcpyHostToDevice));

  // rank(indices) = 3, and m = indices.dim_size(-1)
  TensorShape indices_shape{23, 2};
  TensorTypeDesc indices_desc =
      TensorTypeDesc(indices_shape, PrimitiveType::INT64, place);
  Tensor indices(indices_desc, allocator);

  int64_t* ids = (int64_t*)malloc(indices.numel() * sizeof(int64_t));
  for (size_t i = 0; i < indices.dim_size(0); ++i) {
    ids[i * indices.dim_size(1)] =
        std::experimental::randint(0L, data_shape.dim_size(0) - 1);
    ids[i * indices.dim_size(1) + 1] =
        std::experimental::randint(0L, data_shape.dim_size(1) - 1);
  }

  CudaCheck(cudaMemcpy(indices.mutable_data<int64_t>(), ids,
                       indices.numel() * sizeof(int64_t),
                       cudaMemcpyHostToDevice));

  int64_t slice_size = 1;
  int64_t end_size = indices.dims().back();
  for (int64_t i = end_size; i < data.ndim(); ++i)
    slice_size *= data.dim_size(i);

  auto shape = computeOutputShape(indices.dims(), data.dims());
  TensorShape out_shape(shape);
  TensorTypeDesc out_desc =
      TensorTypeDesc(out_shape, PrimitiveType::INT32, place);

  // rank(output) = rank(data) + rank(indices) - m - 1
  Tensor output(out_desc, allocator);
  output.data<int>();

  ops::GatherNdOp<GPUContext, CUDAPlace, int> gatherNd;
  gatherNd(context, output, data, indices);

  int* cpu_output = (int*)malloc(output.numel() * sizeof(int));
  CudaCheck(cudaMemcpy(cpu_output, output.data<int>(),
                       output.numel() * sizeof(int), cudaMemcpyDeviceToHost));

  checkGatherNdResults<int>(data.dims(), ids, indices.dims(), cpu_output,
                            slice_size);

  free(cpu_data);
  free(ids);
  free(cpu_output);
}

TEST(TestScatterNd, test) {
  cudaStream_t stream;
  CudaCheck(cudaStreamCreate(&stream));

  auto allocator = std::make_shared<CudaMemoryPool>();
  allocator->add_track_stream(stream);

  ops::FillOp<GPUContext, CUDAPlace, float> f;

  CUDAPlace place = CUDAPlace(0);
  GPUContext context(place);

  TensorShape data_shape{177, 257, 213, 137};  // rank(data) = 3
  TensorTypeDesc data_desc =
      TensorTypeDesc(data_shape, PrimitiveType::FLOAT32, place);
  // 177 * 257 * 213 * 137
  Tensor data(data_desc, allocator);
  f(data, 5.);

  // rank(indices) = 3, and m = indices.dim_size(-1)
  TensorShape indices_shape{117, 2};
  TensorTypeDesc indices_desc =
      TensorTypeDesc(indices_shape, PrimitiveType::INT64, place);
  // 117 * 2
  Tensor indices(indices_desc, allocator);

  int64_t* ids = (int64_t*)malloc(indices.numel() * sizeof(int64_t));
  for (size_t i = 0; i < indices.dim_size(0); ++i) {
    ids[i * indices.dim_size(1)] =
        std::experimental::randint(0L, data_shape.dim_size(0) - 1);
    ids[i * indices.dim_size(1) + 1] =
        std::experimental::randint(0L, data_shape.dim_size(1) - 1);
  }

  CudaCheck(cudaMemcpy(indices.mutable_data<int64_t>(), ids,
                       indices.numel() * sizeof(int64_t),
                       cudaMemcpyHostToDevice));

  int64_t slice_size = 1;
  int64_t end_size = indices.dims().back();
  for (int64_t i = end_size; i < data.ndim(); ++i)
    slice_size *= data.dim_size(i);

  auto shape = computeOutputShape(indices.dims(), data.dims());
  TensorShape updates_shape(shape);
  TensorTypeDesc updates_desc =
      TensorTypeDesc(updates_shape, PrimitiveType::FLOAT32, place);
  Tensor updates(updates_desc, allocator);
  f(updates);

  ops::ScatterNdAddOp<GPUContext, CUDAPlace, float> scatterNd;
  scatterNd(context, data, updates, indices);

  float* cpu_output = (float*)malloc(data.numel() * sizeof(float));
  CudaCheck(cudaMemcpy(cpu_output, data.data<float>(),
                       data.numel() * sizeof(float), cudaMemcpyDeviceToHost));
  float* cpu_updates = (float*)malloc(updates.numel() * sizeof(float));
  CudaCheck(cudaMemcpy(cpu_updates, updates.data<float>(),
                       updates.numel() * sizeof(float),
                       cudaMemcpyDeviceToHost));

  checkScatterNdResults<float>(cpu_output, data.dims(), ids, indices.dims(),
                               cpu_updates, slice_size);

  free(ids);
}

}  // namespace core
}  // namespace kaleido
