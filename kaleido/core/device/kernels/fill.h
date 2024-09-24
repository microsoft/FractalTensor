#pragma once

#include "kaleido/core/device/kernels/curand_fp16.h"

#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>

namespace kaleido {
namespace core {
namespace cuda_kernel {

/// Generate random values from uniform distribution
template <typename T>
void FillRandomValue(T* data, int64_t numel);

/// Partial specialization for float.
template <>
void FillRandomValue(float* data, int64_t numel) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    curandGenerateUniform(prng, data, numel);
}

/// Partial specialization for __half
template <>
void FillRandomValue(__half* data, int64_t numel) {
    constexpr auto rng = CURAND_RNG_PSEUDO_XORWOW;
    curand_fp16::generator_t generator;
    curand_fp16::Create(generator, rng);
    curand_fp16::SetSeed(generator, 0);
    curand_fp16::Uniform(generator, data, numel);
    curand_fp16::Destroy(generator);
}

template <>
void FillRandomValue(cutlass::half_t* data, int64_t numel) {
    FillRandomValue<__half>(reinterpret_cast<__half*>(data), numel);
}

/// Generate random values from the normal distribution.
template <typename T>
void FillRandomValue(T* data, int64_t num, float mean, float stddev);

template <>
void FillRandomValue(float* data, int64_t num, float mean, float stddev) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    curandGenerateNormal(prng, data, num, mean, stddev);
}

template <>
void FillRandomValue(__half* data, int64_t numel, float mean, float stddev) {
    constexpr auto rng = CURAND_RNG_PSEUDO_XORWOW;
    curand_fp16::generator_t generator;
    curand_fp16::Create(generator, rng);
    curand_fp16::SetSeed(generator, 0);
    curand_fp16::Normal(generator, data, numel, mean, stddev);
    curand_fp16::Destroy(generator);
}

template <>
void FillRandomValue(cutlass::half_t* data, int64_t numel, float mean,
                     float stddev) {
    FillRandomValue<__half>(reinterpret_cast<__half*>(data), numel, mean,
                            stddev);
}

template <typename T>
__global__ void KeFillValue(T* data, int num, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) data[tid] = static_cast<T>(value);
}

template <>
__global__ void KeFillValue(__half* data, int numel, float value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        data[tid] = __float2half(value);
    }
}

template <>
__global__ void KeFillValue(cutlass::half_t* data, int numel, float value) {
    __half* tmp = reinterpret_cast<__half*>(data);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        tmp[tid] = __float2half(value);
    }
}

template <typename T>
__global__ void KeFillSequential(T* data, int num, float scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num) data[tid] = static_cast<T>(tid * scale);
}

template <>
__global__ void KeFillSequential(__half* data, int numel, float scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        data[tid] = __float2half(tid * scale);
    }
}

template <>
__global__ void KeFillSequential(cutlass::half_t* data, int numel,
                                 float scale) {
    __half* tmp = reinterpret_cast<__half*>(data);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numel) {
        tmp[tid] = __float2half(tid * scale);
    }
}

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
