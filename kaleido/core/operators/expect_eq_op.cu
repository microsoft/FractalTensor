#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/expect_eq_op.h"

#include <cutlass/numeric_types.h>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace kaleido {
namespace core {
namespace ops {

namespace {

template <typename T>
void CompareDeviceData(const T* data1, const T* data2, int64_t numel,
                       float epsilon);

template <typename T>
void CompareDeviceData(const T* data1, const T* data2, int64_t numel,
                       float epsilon) {
    T* h_x1 = (T*)malloc(numel * sizeof(T));
    CudaCheck(
        cudaMemcpy(h_x1, data1, numel * sizeof(T), cudaMemcpyDeviceToHost));

    T* h_x2 = (T*)malloc(numel * sizeof(T));
    CudaCheck(
        cudaMemcpy(h_x2, data2, numel * sizeof(T), cudaMemcpyDeviceToHost));

    bool success = true;
    std::stringstream err_msg;
    for (size_t i = 0; i < numel; ++i) {
#ifdef DEBUG
        std::cout << "Values[" << i << "]: " << h_x1[i] << " vs. " << h_x2[i]
                  << std::endl;
#endif
        if (fabs(h_x1[i] - h_x2[i]) > epsilon) {
            success = false;
            err_msg << "Unequal values[" << i << "]: " << h_x1[i] << " vs. "
                    << h_x2[i] << std::endl;
            break;
        }
    }

    free(h_x1);
    free(h_x2);

    if (!success) {
        throw std::invalid_argument(err_msg.str());
    }
}

__global__ void ConvertFp16ToFp32(float* out, const __half* in, int64_t numel) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numel) {
        out[tid] = __half2float(in[tid]);
    }
}

template <>
void CompareDeviceData(const __half* data1, const __half* data2, int64_t numel,
                       float epsilon) {
    float *fdata1, *fdata2;
    CudaCheck(cudaMalloc(&fdata1, numel * sizeof(float)));
    CudaCheck(cudaMalloc(&fdata2, numel * sizeof(float)));

    const int threads = 128;
    int blocks = DIVUP(numel, threads);

    ConvertFp16ToFp32<<<blocks, threads>>>(fdata1, data1, numel);
    ConvertFp16ToFp32<<<blocks, threads>>>(fdata2, data2, numel);

    float* h_data1 = (float*)malloc(numel * sizeof(float));
    CudaCheck(cudaMemcpy(h_data1, fdata1, numel * sizeof(float),
                         cudaMemcpyDeviceToHost));
    float* h_data2 = (float*)malloc(numel * sizeof(float));
    CudaCheck(cudaMemcpy(h_data2, fdata2, numel * sizeof(float),
                         cudaMemcpyDeviceToHost));

    bool success = true;
    std::stringstream err_msg;
    for (size_t i = 0; i < numel; ++i) {
#ifdef DEBUG
        std::cout << "Values[" << i << "]: " << h_data1[i] << " vs. "
                  << h_data2[i] << std::endl;
#endif
        if (fabs(h_data1[i] - h_data2[i]) > epsilon) {
            success = false;
            err_msg << "Unequal values[" << i << "]: " << h_data1[i] << " vs. "
                    << h_data2[i] << std::endl;
            break;
        }
    }

    CudaCheck(cudaFree(fdata1));
    CudaCheck(cudaFree(fdata2));
    free(h_data1);
    free(h_data2);

    if (!success) {
        throw std::invalid_argument(err_msg.str());
    }
}

template <>
void CompareDeviceData(const cutlass::half_t* data1,
                       const cutlass::half_t* data2, int64_t numel,
                       float epsilon) {
    CompareDeviceData<__half>(reinterpret_cast<const __half*>(data1),
                              reinterpret_cast<const __half*>(data2), numel,
                              epsilon);
}
}  // namespace

template <typename T>
class ExpectEqOp<GPUContext, CUDAPlace, T> {
   public:
    void operator()(const Tensor& x, const Tensor& y, float epsilon = 1e-5) {
        CompareDeviceData<T>(x.data<T>(), y.data<T>(), x.numel(), epsilon);
    }
};

template class ExpectEqOp<GPUContext, CUDAPlace, float>;
template class ExpectEqOp<GPUContext, CUDAPlace, __half>;
template class ExpectEqOp<GPUContext, CUDAPlace, cutlass::half_t>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
