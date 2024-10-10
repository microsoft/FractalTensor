// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/device/cuda_info.h"

#include <gtest/gtest.h>

#include <iostream>

namespace kaleido {
namespace core {

TEST(test, TEST_GET_CUDA_DEVICE_INFO) {
    std::cout << "cuda device count: " << GetGPUDeviceCount() << std::endl;
    std::cout << "Compute Capability: " << GetGPUComputeCapability(0)
              << std::endl;
    std::cout << "Multiprocessors: " << GetGPUMultiProcessors(0) << std::endl;
    std::cout << "Max threads per MP: " << GetGPUMaxThreadsPerMultiProcessor(0)
              << std::endl;
    std::cout << "Max threads per blocks: " << GetGPUMaxThreadsPerBlock(0)
              << std::endl;
    auto grid_size = GetGpuMaxGridDimSize(0);
    std::cout << "Max grid size (x, y, z): " << grid_size.x << ", "
              << grid_size.y << ", " << grid_size.z << std::endl;
}

}  // namespace core
}  // namespace kaleido
