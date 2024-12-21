// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/device/cuda_info.h"

#include "kaleido/core/device/cuda_utils.h"

#include <cuda_runtime.h>

#include <sstream>
#include <vector>

namespace kaleido {
namespace core {

int GetGPUDeviceCount() {
  int deviceCount = 0;
  CudaCheck(cudaGetDeviceCount(&deviceCount));
  return deviceCount;
}

int GetGPUComputeCapability(int id) {
  int major, minor;
  CudaCheck(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id));
  CudaCheck(
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id));
  return major * 10 + minor;
}

int GetGPUMultiProcessors(int id) {
  int count;
  CudaCheck(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, id));
  return count;
}

int GetGPUMaxThreadsPerMultiProcessor(int id) {
  int count;
  CudaCheck(cudaDeviceGetAttribute(&count,
                                   cudaDevAttrMaxThreadsPerMultiProcessor, id));
  return count;
}

int GetGPUMaxThreadsPerBlock(int id) {
  int count;
  CudaCheck(cudaDeviceGetAttribute(&count, cudaDevAttrMaxThreadsPerBlock, id));
  return count;
}

dim3 GetGpuMaxGridDimSize(int id) {
  dim3 grid_size;

  int size;
  CudaCheck(cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimX, id));
  grid_size.x = size;

  CudaCheck(cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimY, id));
  grid_size.y = size;

  CudaCheck(cudaDeviceGetAttribute(&size, cudaDevAttrMaxGridDimZ, id));
  grid_size.z = size;
  return grid_size;
}

std::string GetDeviceName() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::stringstream ss(prop.name);
  const char delim = ' ';

  std::string s;
  std::vector<std::string> out;

  while (std::getline(ss, s, delim)) {
    out.push_back(s);
  }

  std::stringstream out_ss;
  int i = 0;
  for (; i < out.size() - 1; ++i) out_ss << out[i] << "_";
  out_ss << out[i];
  return out_ss.str();
}

}  // namespace core
}  // namespace kaleido
