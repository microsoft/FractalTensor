#pragma once
#include <cuda_runtime.h>

#include <string>
namespace kaleido {
namespace core {

int GetGPUDeviceCount();

int GetGPUComputeCapability(int id);

int GetGPUMultiProcessors(int id);

int GetGPUMaxThreadsPerMultiProcessor(int id);

int GetGPUMaxThreadsPerBlock(int id);

dim3 GetGpuMaxGridDimSize(int);

std::string GetDeviceName();

}  // namespace core
}  // namespace kaleido
