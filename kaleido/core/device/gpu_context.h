#pragma once

#include "kaleido/core/device/cuda_info.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/device_context.h"
#include "kaleido/core/place.h"

namespace kaleido {
namespace core {

class GPUContext : public DeviceContext {
   public:
    GPUContext();
    explicit GPUContext(const CUDAPlace& place) : place_{place} {
        CublasCheck(cublasCreate(&cublas_handle_));
        CublasCheck(
            cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_HOST));
        CudnnCheck(cudnnCreate(&cudnn_handle_));

        compute_capability_ = GetGPUComputeCapability(place_.GetDeviceId());
        multi_process_ = GetGPUMultiProcessors(place_.GetDeviceId());
        max_threads_per_mp_ =
            GetGPUMaxThreadsPerMultiProcessor(place_.GetDeviceId());
        max_threads_per_block_ = GetGPUMaxThreadsPerBlock(place_.GetDeviceId());
        max_grid_dim_size_ = GetGpuMaxGridDimSize(place_.GetDeviceId());
        device_name_ = GetDeviceName();
    }

    static GPUContext& GetInstance() {
        static GPUContext context;
        return context;
    }

    ~GPUContext();
    GPUContext(GPUContext const&) = delete;
    void operator=(GPUContext const&) = delete;

    int GetComputeCapability() const { return compute_capability_; };

    int GetMaxPhysicalThreadCount() const {
        return multi_process_ * max_threads_per_mp_;
    };

    int GetMaxThreadsPerBlock() const { return max_threads_per_block_; };

    int GetSMCount() const { return multi_process_; };

    dim3 GetCUDAMaxGridDimSize() const { return max_grid_dim_size_; };
    std::string GetDeviceName() const { return device_name_; }

    cublasHandle_t cublas_handle() const { return cublas_handle_; }
    cudnnHandle_t cudnn_handle() const { return cudnn_handle_; }

   private:
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;

    CUDAPlace place_;
    int compute_capability_;
    int multi_process_;
    int max_threads_per_mp_;
    int max_threads_per_block_;
    dim3 max_grid_dim_size_;
    std::string device_name_;
};

}  // namespace core
}  // namespace kaleido
