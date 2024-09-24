#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/print_op.h"

#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include <glog/logging.h>

#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

template <typename T>
std::string ToString(const T* data, int64_t numel, int64_t stride,
                     int precision) {
    std::ostringstream value_str;
    value_str.setf(std::ios::fixed);
    value_str << std::setprecision(precision);

    value_str << "0\t";
    for (size_t i = 0; i < numel; ++i) {
        value_str << static_cast<T>(data[i]) << " ";
        if (stride > 0 && i && (i + 1) % stride == 0)
            value_str << std::endl << (i + 1) / stride << "\t";
    }
    value_str << std::endl;
    return value_str.str();
}

template <typename T>
std::string GetValueStr(const T* src, int64_t numel, int64_t stride,
                        int precision);

template <typename T>
std::string GetValueStr(const T* src, int64_t numel, int64_t stride,
                        int precision) {
    T* hTmp = (T*)malloc(numel * sizeof(T));
    if (!hTmp) LOG(FATAL) << "fail to allocate memory.";
    kaleido::core::CudaCheck(
        cudaMemcpy(hTmp, src, numel * sizeof(T), cudaMemcpyDeviceToHost));

    std::string value_str = ToString<T>(hTmp, numel, stride, precision);
    free(hTmp);
    return value_str;
};

__global__ void ConvertFp16ToFp32(float* out, const __half* in, int64_t numel) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numel) {
        out[tid] = __half2float(in[tid]);
    }
}

template <>
std::string GetValueStr(const __half* src, int64_t numel, int64_t stride,
                        int precision) {
    const int threads = 64;
    int blocks = DIVUP(numel, threads);

    float* dTmp;
    kaleido::core::CudaCheck(cudaMalloc(&dTmp, numel * sizeof(float)));
    ConvertFp16ToFp32<<<blocks, threads>>>(dTmp, src, numel);

    float* hTmp = (float*)malloc(numel * sizeof(float));
    if (!hTmp) LOG(FATAL) << "fail to allocate memory.";
    kaleido::core::CudaCheck(
        cudaMemcpy(hTmp, dTmp, numel * sizeof(float), cudaMemcpyDeviceToHost));

    std::string value_str = ToString<float>(hTmp, numel, stride, precision);

    kaleido::core::CudaCheck(cudaFree(dTmp));
    free(hTmp);

    return value_str;
}

template <>
std::string GetValueStr(const cutlass::half_t* src, int64_t numel,
                        int64_t stride, int precision) {
    return GetValueStr<__half>(reinterpret_cast<const __half*>(src), numel,
                               stride, precision);
}

}  // namespace

namespace kaleido {
namespace core {
namespace ops {

template <typename T>
class PrintOp<GPUContext, CUDAPlace, T> {
   public:
    std::string operator()(const Tensor& input, int precision = 3,
                           int count = -1, int pos = -1) const {
        int offset = pos > 0 ? pos : 0;
        int num = count > 0 ? count : input.numel();
        int stride = num > input.dim_size(-1) ? input.dim_size(-1) : num;
        return GetValueStr<T>(input.data<T>() + offset, num, stride, precision);
    }
};

template class PrintOp<GPUContext, CUDAPlace, float>;
template class PrintOp<GPUContext, CUDAPlace, __half>;
template class PrintOp<GPUContext, CUDAPlace, cutlass::half_t>;
template class PrintOp<GPUContext, CUDAPlace, int>;
template class PrintOp<GPUContext, CUDAPlace, int64_t>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
