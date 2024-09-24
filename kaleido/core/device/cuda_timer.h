#pragma once

#include "kaleido/core/device/cuda_utils.h"

namespace kaleido {
namespace core {
class CudaTimer {
   private:
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaStream_t stream;

   public:
    CudaTimer() {
        CudaCheck(cudaEventCreate(&start));
        CudaCheck(cudaEventCreate(&stop));
    }

    ~CudaTimer() {
        CudaCheck(cudaEventDestroy(start));
        CudaCheck(cudaEventDestroy(stop));
    }

    void Start(cudaStream_t st = 0) {
        stream = st;
        CudaCheck(cudaEventRecord(start, stream));
    }

    float Stop() {
        float milliseconds = 0.;
        CudaCheck(cudaEventRecord(stop, stream));
        CudaCheck(cudaEventSynchronize(stop));
        CudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
};
}  // namespace core
}  // namespace kaleido
