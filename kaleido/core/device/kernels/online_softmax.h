#pragma once

#include "kaleido/core/device/kernels/reduce.h"

namespace kaleido {
namespace core {
namespace cuda_kernel {

#define THRESHOLD 64

template <typename T>
__device__ void Power2ReduceMD(T& val_m, T& val_d, int tid, int block_size,
                               T* shm, T* shd) {
    unsigned mask = 0u;

    bool cur_bigger;
    T next_val_m;
    T next_val_d;

    CREATE_SHFL_MASK(mask, tid < block_size);

    next_val_m = CudaShuffleDownSync(mask, val_m, 16);
    next_val_d = CudaShuffleDownSync(mask, val_d, 16);
    cur_bigger = (val_m > next_val_m);
    val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                       : next_val_d + val_d * exp(val_m - next_val_m);
    val_m = cur_bigger ? val_m : next_val_m;

    next_val_m = CudaShuffleDownSync(mask, val_m, 8);
    next_val_d = CudaShuffleDownSync(mask, val_d, 8);
    cur_bigger = (val_m > next_val_m);
    val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                       : next_val_d + val_d * exp(val_m - next_val_m);
    val_m = cur_bigger ? val_m : next_val_m;

    next_val_m = CudaShuffleDownSync(mask, val_m, 4);
    next_val_d = CudaShuffleDownSync(mask, val_d, 4);
    cur_bigger = (val_m > next_val_m);
    val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                       : next_val_d + val_d * exp(val_m - next_val_m);
    val_m = cur_bigger ? val_m : next_val_m;

    next_val_m = CudaShuffleDownSync(mask, val_m, 2);
    next_val_d = CudaShuffleDownSync(mask, val_d, 2);
    cur_bigger = (val_m > next_val_m);
    val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                       : next_val_d + val_d * exp(val_m - next_val_m);
    val_m = cur_bigger ? val_m : next_val_m;

    next_val_m = CudaShuffleDownSync(mask, val_m, 1);
    next_val_d = CudaShuffleDownSync(mask, val_d, 1);
    cur_bigger = (val_m > next_val_m);
    val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                       : next_val_d + val_d * exp(val_m - next_val_m);
    val_m = cur_bigger ? val_m : next_val_m;

    if (tid < warpSize) {
        shm[tid] = -1.e20;
        shd[tid] = 1.0F;
    }
    __syncthreads();

    if (tid % warpSize == 0) {
        shm[tid / warpSize] = val_m;
        shd[tid / warpSize] = val_d;
    }
    __syncthreads();

    CREATE_SHFL_MASK(mask, tid < warpSize);
    if (tid < warpSize) {
        val_m = shm[tid];
        val_d = shd[tid];

        next_val_m = CudaShuffleDownSync(mask, val_m, 16);
        next_val_d = CudaShuffleDownSync(mask, val_d, 16);
        cur_bigger = (val_m > next_val_m);
        val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                           : next_val_d + val_d * exp(val_m - next_val_m);
        val_m = cur_bigger ? val_m : next_val_m;

        next_val_m = CudaShuffleDownSync(mask, val_m, 8);
        next_val_d = CudaShuffleDownSync(mask, val_d, 8);
        cur_bigger = (val_m > next_val_m);
        val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                           : next_val_d + val_d * exp(val_m - next_val_m);
        val_m = cur_bigger ? val_m : next_val_m;

        next_val_m = CudaShuffleDownSync(mask, val_m, 4);
        next_val_d = CudaShuffleDownSync(mask, val_d, 4);
        cur_bigger = (val_m > next_val_m);
        val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                           : next_val_d + val_d * exp(val_m - next_val_m);
        val_m = cur_bigger ? val_m : next_val_m;

        next_val_m = CudaShuffleDownSync(mask, val_m, 2);
        next_val_d = CudaShuffleDownSync(mask, val_d, 2);
        cur_bigger = (val_m > next_val_m);
        val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                           : next_val_d + val_d * exp(val_m - next_val_m);
        val_m = cur_bigger ? val_m : next_val_m;

        next_val_m = CudaShuffleDownSync(mask, val_m, 1);
        next_val_d = CudaShuffleDownSync(mask, val_d, 1);
        cur_bigger = (val_m > next_val_m);
        val_d = cur_bigger ? val_d + next_val_d * exp(next_val_m - val_m)
                           : next_val_d + val_d * exp(val_m - next_val_m);
        val_m = cur_bigger ? val_m : next_val_m;
    }
}

template <typename T>
__device__ __forceinline__ void ReduceMD(const T* I, T* shm, T* shd,
                                         int block_size, int base, int cur_idx,
                                         int next_idx, int width) {
    T val_m = -1.e20;
    T val_d = 1.0F;
    while (cur_idx < width) {
        if (val_m < I[next_idx]) val_m = I[next_idx];
        if (cur_idx != base) {
            if (val_m < I[next_idx]) {
                val_d = 1 + val_d * exp(val_m - I[next_idx]);
                val_m = I[next_idx];
            } else {
                val_d = val_d + exp(I[next_idx] - val_m);
            }
        }
        next_idx += block_size;
        cur_idx += block_size;
    }
    __syncthreads();

    Power2ReduceMD(val_m, val_d, base, block_size, shm, shd);

    if (0 == base) {
        shm[0] = val_m;
        shd[0] = val_d;
    }

    __syncthreads();
}

template <typename T>
__device__ __forceinline__ void MapRescale(const T* I, T* O, T max, T sum,
                                           int block_size, int base,
                                           int cur_idx, int next_idx,
                                           int width) {
    float val = 0.;
    while (cur_idx < width) {
        val = I[next_idx] - max;
        if (val < -THRESHOLD) val = -THRESHOLD;
        O[next_idx] = exp(val) / sum;

        next_idx += block_size;
        cur_idx += block_size;
    }
    __syncthreads();
}

template <typename T>
__device__ __forceinline__ void OnlineSoftmax(const T* I, T* O, int block_size,
                                              int base, int cur_idx,
                                              int next_idx, int width) {
    const int warpSize = 32;
    __shared__ T shm[warpSize];
    __shared__ T shd[warpSize];

    ReduceMD(I, shm, shd, block_size, base, cur_idx, next_idx, width);

    MapRescale(I, O, shm[0], shd[0], block_size, base, cur_idx, next_idx,
               width);
}

template <typename T>
__global__ void KeOnlineNormalizedSoftMax(const T* I, T* O,
                                          const size_t width) {
    int block_size = blockDim.x;
    int base = threadIdx.x;
    int next_idx = blockIdx.x * width + base;
    int cur_idx = base;
    OnlineSoftmax(I, O, block_size, base, cur_idx, next_idx, width);
}

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
