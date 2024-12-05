#pragma once

#include "kaleido/core/device/cuda_utils.h"

namespace kaleido {
namespace core {
namespace cuda_kernel {

template <typename T, typename Functor>
__device__ __forceinline__ void ElementwiseUnary(T* out, const T* in,
                                                 Functor op, int offset) {
    out[offset] = op(in[offset]);
}

template <typename T, typename Functor>
__device__ __forceinline__ void ElementwiseBinary(T* out, const T* in1,
                                                  const T* in2, Functor op,
                                                  int offset) {
    out[offset] = op(in1[offset], in2[offset]);
}

template <typename T, typename Functor>
__device__ __forceinline__ void ElementwiseTernary(T* out, const T* in1,
                                                   const T* in2, const T* in3,
                                                   Functor op, int offset) {
    out[offset] = op(in1[offset], in2[offset], in3[offset]);
}

template <typename T, typename Functor>
__device__ __forceinline__ void ElementwiseArityFour(T* out, const T* in1,
                                                     const T* in2, const T* in3,
                                                     const T* in4, Functor op,
                                                     int offset) {
    out[offset] = op(in1[offset], in2[offset], in3[offset], in4[offset]);
}

template <typename T, typename Functor>
__device__ __forceinline__ void ElementwiseAny(T* out, const T** ins,
                                               Functor op, int offset,
                                               int num_inputs) {
    T args[num_inputs];
#pragma unroll
    for (int i = 0; i < num_inputs; ++i) {
        args[i] = ins[i][offset];
    }
    out[offset] = op(args);
}

template <typename T, typename Functor, int Arity>
struct ElementwiseCaller {
    __device__ inline void operator()(Functor func, const T** inputs, T* output,
                                      int offset, int num_inputs);
};

template <typename T, typename Functor>
struct ElementwiseCaller<T, Functor, 1> {
    __device__ inline void operator()(Functor func, const T** inputs, T* output,
                                      int offset, int num_inputs) {
        ElementwiseUnary<T, Functor>(output, inputs[0], func, offset);
    }
};

template <typename T, typename Functor>
struct ElementwiseCaller<T, Functor, 2> {
    __device__ inline void operator()(Functor func, const T** inputs, T* output,
                                      int offset, int num_inputs) {
        ElementwiseBinary<T, Functor>(output, inputs[0], inputs[1], func,
                                      offset);
    }
};

template <typename T, typename Functor>
struct ElementwiseCaller<T, Functor, 3> {
    __device__ inline void operator()(Functor func, const T** inputs, T* output,
                                      int offset, int num_inputs) {
        ElementwiseTernary<T, Functor>(output, inputs[0], inputs[1], inputs[2],
                                       func, offset);
    }
};

template <typename T, typename Functor>
struct ElementwiseCaller<T, Functor, -1> {
    __device__ inline void operator()(Functor func, const T** inputs, T* output,
                                      int offset, int num_inputs) {
        ElementwiseAny<T, Functor>(output, inputs, func, offset, num_inputs);
    }
};

template <typename T, typename Functor, int Arity>
__global__ void ElementwiseKernel(const T** inputs, T* output, int numel,
                                  Functor func, int num_inputs) {
    int cur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cur_idx < numel) {
        ElementwiseCaller<T, Functor, Arity>()(func, inputs, output, cur_idx,
                                               num_inputs);
    }
}

template <typename T, typename Functor>
__global__ void ElementwiseUnaryKernel(const T* in, T* out, int numel,
                                       Functor func) {
    int block_size = blockDim.x;
    int cur_idx = threadIdx.x;

    for (; cur_idx < numel; cur_idx += block_size) {
        ElementwiseUnary<T, Functor>(out, in, func, cur_idx);
    }
}

template <typename T, typename Functor>
__global__ void ElementwiseBinaryKernel(const T* in1, const T* in2, T* out,
                                        int numel, Functor func) {
    int cur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cur_idx < numel)
        ElementwiseBinary<T, Functor>(out, in1, in2, func, cur_idx);
}

template <typename T, typename Functor>
__global__ void ElementwiseTernaryKernel(const T* in1, const T* in2,
                                         const T* in3, T* out, int numel,
                                         Functor func) {
    int cur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cur_idx < numel)
        ElementwiseTernary<T, Functor>(out, in1, in2, in3, func, cur_idx);
}

template <typename T, typename Functor>
__global__ void ElementwiseArityFourKernel(const T* in1, const T* in2,
                                           const T* in3, const T* in4, T* out,
                                           int numel, Functor func) {
    int cur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cur_idx < numel)
        ElementwiseArityFour<T, Functor>(out, in1, in2, in3, in4, func,
                                         cur_idx);
}

// FIXME(ying): Simulate fused LSTM activations. Re-implement this
// kernel.
template <typename T>
__global__ void ElementwiseLstmAct(const T* igx, const T* igu, const T* fgx,
                                   const T* fgu, const T* ogx, const T* ogu,
                                   const T* cgx, const T* cgu, const T* c,
                                   T* out1, T* out2, int numel) {
    int cur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cur_idx < numel) {
        T i_act = 1. / (1. + exp(-igx[cur_idx] - igu[cur_idx]));
        T f_act = 1. / (1. + exp(-fgx[cur_idx] - fgu[cur_idx]));
        T o_act = 1. / (1. + exp(-ogx[cur_idx] - ogx[cur_idx]));
        T c_act = tanh(cgx[cur_idx] + cgu[cur_idx]);
        out1[cur_idx] = f_act * c[cur_idx] + i_act * c_act;
        out2[cur_idx] = o_act * tanh(out1[cur_idx]);
    }
}

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
