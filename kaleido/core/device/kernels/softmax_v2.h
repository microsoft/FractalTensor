#pragma once

#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/operators/kernels/math_functor.h"
#include "kaleido/core/operators/kernels/reduce.h"
#include "kaleido/core/operators/kernels/softmax_common.h"

namespace kaleido {
namespace core {
namespace cuda_kernel {

template <typename T, int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void KeMatrixSoftMaxV2(const T* __restrict input, T* __restrict output,
                           int width) {
    // use shared memory to cache input and intermediate results.
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* sbuf = reinterpret_cast<T*>(shared_buf);
    __shared__ MD<T> md;

    // cache input into shared memory
    int tid = threadIdx.x;
    int next_idx = blockIdx.x * width + tid;  // element index in input array
    int cur_idx = tid;                        // element index in current row
    for (; cur_idx < width; next_idx += BLOCK_SIZE, cur_idx += BLOCK_SIZE) {
        sbuf[cur_idx] = input[next_idx];
    }
    __syncthreads();

    // Loop1: reduction Max, the maximum value is stored in md.m.
    Max<T> max;
    md.m =
        BlockRowReduce<T, Max<T>, BLOCK_SIZE>(sbuf, width, max, -MaxValue<T>());
    __syncthreads();

    // Loop2: reduction sum of exponential and substraction:
    // sum(exp(x - m)). the reduction sum is stored in md.d;
    SubAndExp<T> sub_and_exp(md.m);  // mapper
    Add<T> sum;                      // reducer
    Inverse<T> inverse;              // finalizer
    md.d = BlockRowReduce<T, SubAndExp<T>, Add<T>, Inverse<T>, BLOCK_SIZE>(
        sbuf, sbuf, width, sub_and_exp /*mapper*/, sum /*reducer*/,
        inverse /*finalizer*/, static_cast<T>(0) /*initialier of reduction*/);
    __syncthreads();

    // Loop3: map to rescale.
    for (int cur_idx = tid; cur_idx < width; cur_idx += BLOCK_SIZE) {
        sbuf[cur_idx] *= md.d;
    }

    // Store result into global memory.
    tid = threadIdx.x;
    next_idx = blockIdx.x * width + tid;
    cur_idx = tid;
    for (; cur_idx < width; next_idx += BLOCK_SIZE, cur_idx += BLOCK_SIZE) {
        output[next_idx] = sbuf[cur_idx];
    }
}

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
