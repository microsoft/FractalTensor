#pragma once

#include "kaleido/core/device/kernels/math_functor.h"
#include "kaleido/core/device/kernels/reduce.h"

namespace kaleido {
namespace core {
namespace cuda_kernel {

#define THRESHOLD 64

template <typename T>
__device__ T ReduceMax(float val, int tid, int block_size, T* shm) {
    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, tid < block_size);

    val = max(val, CudaShuffleDownSync(mask, val, 16));
    val = max(val, CudaShuffleDownSync(mask, val, 8));
    val = max(val, CudaShuffleDownSync(mask, val, 4));
    val = max(val, CudaShuffleDownSync(mask, val, 2));
    val = max(val, CudaShuffleDownSync(mask, val, 1));

    if (tid < warpSize) shm[tid] = 0.;
    __syncthreads();

    if (tid % warpSize == 0) shm[tid / warpSize] = val;
    __syncthreads();

    CREATE_SHFL_MASK(mask, tid < warpSize);

    if (tid < warpSize) {
        val = shm[tid];

        val = max(val, CudaShuffleDownSync(mask, val, 16));
        val = max(val, CudaShuffleDownSync(mask, val, 8));
        val = max(val, CudaShuffleDownSync(mask, val, 4));
        val = max(val, CudaShuffleDownSync(mask, val, 2));
        val = max(val, CudaShuffleDownSync(mask, val, 1));
    }

    return val;
}

template <typename T>
__device__ __forceinline__ void FindMax(const T* I, T* shm, int block_size,
                                        int base, int cur_idx, int next_idx,
                                        int width) {
    T val = -1.e20;

    while (cur_idx < width) {
        if (val < I[next_idx]) val = I[next_idx];
        next_idx += block_size;
        cur_idx += block_size;
    }
    __syncthreads();

    val = ReduceMax(val, base, block_size, shm);

    if (0 == base) shm[0] = val;

    __syncthreads();
}

template <typename T>
__device__ __forceinline__ void SubMaxAndExp(const T* I, T* O, int cur_idx,
                                             int next_idx, int block_size,
                                             int width, float max) {
    float val = 0.;
    while (cur_idx < width) {
        val = I[next_idx] - max;
        if (val < -THRESHOLD) val = -THRESHOLD;
        O[next_idx] = exp(val);

        next_idx += block_size;
        cur_idx += block_size;
    }
    __syncthreads();
}

template <typename T>
__device__ float ReduceSum(T val, int tid, int block_size, float* shm) {
    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, tid < block_size);

    val += CudaShuffleDownSync(mask, val, 16);
    val += CudaShuffleDownSync(mask, val, 8);
    val += CudaShuffleDownSync(mask, val, 4);
    val += CudaShuffleDownSync(mask, val, 2);
    val += CudaShuffleDownSync(mask, val, 1);

    if (tid < warpSize) shm[tid] = 0.;
    __syncthreads();

    if (tid % warpSize == 0) shm[tid / warpSize] = val;

    __syncthreads();

    CREATE_SHFL_MASK(mask, tid < warpSize);

    if (tid < warpSize) {
        val = shm[tid];

        val += CudaShuffleDownSync(mask, val, 16);
        val += CudaShuffleDownSync(mask, val, 8);
        val += CudaShuffleDownSync(mask, val, 4);
        val += CudaShuffleDownSync(mask, val, 2);
        val += CudaShuffleDownSync(mask, val, 1);
    }

    return val;
}

template <typename T>
__device__ __forceinline__ void ValueSum(T* O, T* shm, int block_size, int base,
                                         int cur_idx, int next_idx, int width) {
    T val = 0.;
    while (cur_idx < width) {
        val += O[next_idx];
        next_idx += block_size;
        cur_idx += block_size;
    }
    __syncthreads();

    val = ReduceSum(val, base, block_size, shm);
    if (base == 0) shm[0] = val;

    __syncthreads();
}

template <typename T>
__device__ __forceinline__ void DivSum(T* O, float sum, int cur_idx,
                                       int next_idx, int block_size,
                                       int width) {
    while (cur_idx < width) {
        O[next_idx] /= sum;
        next_idx += block_size;
        cur_idx += block_size;
    }
}

template <typename T>
__device__ __forceinline__ void Softmax(const T* I, T* O, int block_size,
                                        int base, int cur_idx, int next_idx,
                                        int width) {
    const int warpSize = 32;
    __shared__ T shm[warpSize];

    // find the max number, max value is stored in shm[0]
    FindMax(I, shm, block_size, base, cur_idx, next_idx, width);

    // sub max Value and do Exp operation
    SubMaxAndExp(I, O, base, next_idx, block_size, width, shm[0]);

    // add width values into blockDim.x buffer, sum is in shm[0]
    ValueSum(O, shm, block_size, base, cur_idx, next_idx, width);

    // divided by sum
    DivSum(O, shm[0], cur_idx, next_idx, block_size, width);
}

template <typename T>
__global__ void KeMatrixSoftMax(const T* I, T* O, const size_t width) {
    int block_size = blockDim.x;
    int base = threadIdx.x;
    int next_idx = blockIdx.x * width + base;
    int cur_idx = base;

    Softmax(I, O, block_size, base, cur_idx, next_idx, width);
}

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
