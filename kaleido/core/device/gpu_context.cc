#include "kaleido/core/device/gpu_context.h"

#include "kaleido/core/device/cuda_info.h"

namespace kaleido {
namespace core {

GPUContext::GPUContext() {
    CublasCheck(cublasCreate(&cublas_handle_));
    CublasCheck(cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_HOST));
    // CudnnCheck(cudnnCreate(&cudnn_handle_));

    compute_capability_ = GetGPUComputeCapability(0);
    multi_process_ = GetGPUMultiProcessors(0);
    max_threads_per_mp_ = GetGPUMaxThreadsPerMultiProcessor(0);
    max_threads_per_block_ = GetGPUMaxThreadsPerBlock(0);
    max_grid_dim_size_ = GetGpuMaxGridDimSize(0);

    device_name_ = GetDeviceName();
}

GPUContext::~GPUContext() {
    CublasCheck(cublasDestroy(cublas_handle_));
    // CudnnCheck(cudnnDestroy(cudnn_handle_));
}

}  // namespace core
}  // namespace kaleido
