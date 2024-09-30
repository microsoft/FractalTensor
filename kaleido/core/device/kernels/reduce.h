#pragma once

#include <float.h>

namespace kaleido {
namespace core {
namespace cuda_kernel {

#if CUDA_VERSION < 9000
#define CREATE_SHFL_MASK(mask, predicate) mask = 0u;
#else
#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
    mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#endif

template <typename T>
__inline__ __device__ T MaxValue();

template <>
__inline__ __device__ float MaxValue<float>() {
    return FLT_MAX;
}

template <>
__inline__ __device__ double MaxValue<double>() {
    return DBL_MAX;
}

__forceinline__ __device__ float CudaShuffleDownSync(unsigned mask, float val,
                                                     int delta,
                                                     int width = 32) {
    return __shfl_down_sync(mask, val, delta, width);
}

template <typename T>
__forceinline__ __device__ float CudaShuffleDownSync(unsigned mask, T val,
                                                     int delta,
                                                     int width = 32) {
    return __shfl_down_sync(mask, val, delta, width);
}

template <typename T>
__forceinline__ __device__ T CudaShuffleSync(unsigned mask, T val, int src_line,
                                             int width = 32) {
    return __shfl_sync(mask, val, src_line, width);
}

template <typename T, typename Reducer>
__forceinline__ __device__ T WrapReduce(T val, unsigned mask, Reducer reducer) {
    val = reducer(val, CudaShuffleDownSync(mask, val, 16));
    val = reducer(val, CudaShuffleDownSync(mask, val, 8));
    val = reducer(val, CudaShuffleDownSync(mask, val, 4));
    val = reducer(val, CudaShuffleDownSync(mask, val, 2));
    val = reducer(val, CudaShuffleDownSync(mask, val, 1));
    return val;
}

// Works only for power-of-2 arrays.
template <typename T, typename Reducer>
__device__ T Power2Reduce(T val, int tid, T* __restrict shm, Reducer reducer,
                          T init_val) {
    int block_size = blockDim.x;
    unsigned mask = 0u;
    CREATE_SHFL_MASK(mask, tid < block_size);

    val = WrapReduce(val, mask, reducer);

    if (tid < warpSize) shm[tid] = init_val;
    __syncthreads();

    if (tid % warpSize == 0) shm[tid / warpSize] = val;
    __syncthreads();

    CREATE_SHFL_MASK(mask, tid < warpSize);
    if (tid < warpSize) {
        val = shm[tid];
        val = WrapReduce(val, mask, reducer);
    }
    return val;
}

/*
 * A single thread block computes the reduction of n values.
 */
template <typename T, typename Reducer, int BLOCK_SIZE>
__device__ T BlockRowReduce(const T* __restrict input, int n /*length*/,
                            Reducer reducer, T init_val) {
    T val = init_val;
    int tid = threadIdx.x;
    for (int cur_idx = tid; cur_idx < n; cur_idx += BLOCK_SIZE) {
        val = reducer(val, input[cur_idx]);
    }
    __syncthreads();

    const int kWarpSize = 32;
    __shared__ T shm[kWarpSize];
    val = Power2Reduce(val, tid, shm, reducer, init_val);

    if (threadIdx.x == 0) {
        return val;
    }
}

/*
 * A single thread block computes the reduction of n values.
 * This function computes:
 *   val = init_val
 *   for (i, x) in enumerate(input):
 *       output[i] = mapper(input[i])
 *       val = reducer(val, output[i])
 *   val = finalizer(val)
 *   return val
 */
template <typename T, typename Mapper, typename Reducer, typename Finalizer,
          int BLOCK_SIZE>
__device__ T BlockRowReduce(const T* __restrict input, T* __restrict output,
                            int n /*length*/, Mapper mapper, Reducer reducer,
                            Finalizer finalizer, T init_val) {
    T val = init_val;
    int tid = threadIdx.x;
    for (int cur_idx = tid; cur_idx < n; cur_idx += BLOCK_SIZE) {
        output[cur_idx] = mapper(input[cur_idx]);
        val = reducer(val, output[cur_idx]);
    }
    __syncthreads();

    const int kWarpSize = 32;
    __shared__ T shm[kWarpSize];
    val = Power2Reduce(val, tid, shm, reducer, init_val);

    if (threadIdx.x == 0) {
        return finalizer(val);
    }
}

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
