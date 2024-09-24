#pragma once

#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cstdint>
#include <stdexcept>

namespace kaleido {
namespace core {
namespace cuda_kernel {

namespace curand_fp16 {

namespace {
constexpr unsigned block_size = 256;
constexpr unsigned store_block_batch_size = 1;
constexpr unsigned num_sm_scale = 1;
using block_t = ulong1;

template <class T>
__global__ void StatusInitKernel(T* const status_ptr,
                                 const std::uint64_t seed) {
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tid, 0, status_ptr + tid);
}

template <class T>
struct size_of {
    int value = 0;
};
template <>
struct size_of<ushort1> {
    static const int value = 2;
};
template <>
struct size_of<uint1> {
    static const int value = 4;
};
template <>
struct size_of<ulong1> {
    static const int value = 8;
};
template <>
struct size_of<half> {
    static const int value = 2;
};
template <>
struct size_of<half2> {
    static const int value = 4;
};

template <class RNG_T, class BLOCK_T, uint32_t pm = 0>
__global__ void GenerateKernel(half* const array_ptr, RNG_T* const status_ptr,
                               const std::size_t size) {
    constexpr auto batch_size =
        size_of<BLOCK_T>::value / size_of<half>::value * store_block_batch_size;
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    auto curand_gen = *(status_ptr + tid);

    const auto batch_loop_size = size - (size % batch_size);
    for (unsigned i = tid * batch_size; i < batch_loop_size;
         i += batch_size * gridDim.x * blockDim.x) {
        // block gen
        union {
            half h1[size_of<BLOCK_T>::value / size_of<half>::value];
            half2 h2[size_of<BLOCK_T>::value / size_of<half2>::value];
            BLOCK_T store_block;
            unsigned u[size_of<BLOCK_T>::value / size_of<uint1>::value];
            short s[size_of<BLOCK_T>::value / size_of<ushort1>::value];
        } batch_block[store_block_batch_size];

#pragma unroll
        for (unsigned sb = 0; sb < store_block_batch_size; sb++) {
#pragma unroll
            for (unsigned j = 0;
                 j < size_of<BLOCK_T>::value / size_of<half2>::value; j++) {
                batch_block[sb].u[j] = curand(&curand_gen);
                if constexpr (pm == 0) {
                    batch_block[sb].u[j] &= 0x7fff7fffu;
                }
            }
        }
#pragma unroll
        for (unsigned sb = 0; sb < store_block_batch_size; sb++) {
#pragma unroll
            for (unsigned j = 0;
                 j < size_of<BLOCK_T>::value / size_of<half>::value; j++) {
                batch_block[sb].h1[j] = __short2half_rn(batch_block[sb].s[j]);
            }
        }
#pragma unroll
        for (unsigned sb = 0; sb < store_block_batch_size; sb++) {
#pragma unroll
            for (unsigned j = 0;
                 j < size_of<BLOCK_T>::value / size_of<half2>::value; j++) {
                batch_block[sb].h2[j] = __hmul2(batch_block[sb].h2[j],
                                                __float2half2_rn(1.f / 0x7fff));
            }
        }
#pragma unroll
        for (unsigned sb = 0; sb < store_block_batch_size; sb++) {
            *(reinterpret_cast<BLOCK_T*>(array_ptr + i) + sb) =
                batch_block[sb].store_block;
        }
    }
    if (tid == 0) {
        const auto res = size - batch_loop_size;
        if (res != 0) {
            for (unsigned j = 0; j < res; j++) {
                const auto v = static_cast<short>(curand(&curand_gen) &
                                                  (pm ? 0xffff : 0x7fff));
                array_ptr[batch_loop_size + j] =
                    __hmul(__short2half_rn(v), __float2half_rn(1.f / 0x7fff));
            }
        }
    }
    *(status_ptr + tid) = curand_gen;
}

template <class RNG_T, class BLOCK_T>
__global__ void GenerateNormalKernel(half* const array_ptr,
                                     RNG_T* const status_ptr,
                                     const std::size_t size, const float mean,
                                     const float sigma) {
    constexpr auto batch_size =
        size_of<BLOCK_T>::value / size_of<half>::value * store_block_batch_size;
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
    auto curand_gen = *(status_ptr + tid);

    const auto batch_loop_size = size - (size % batch_size);
    for (unsigned i = tid * batch_size; i < batch_loop_size;
         i += batch_size * gridDim.x * blockDim.x) {
        // block gen
        union {
            half h1[size_of<BLOCK_T>::value / size_of<half>::value];
            half2 h2[size_of<BLOCK_T>::value / size_of<half2>::value];
            BLOCK_T store_block;
            unsigned u[size_of<BLOCK_T>::value / size_of<uint1>::value];
            short s[size_of<BLOCK_T>::value / size_of<ushort1>::value];
            ushort2 s2[size_of<BLOCK_T>::value / size_of<uint1>::value];
        } batch_block[store_block_batch_size];

#pragma unroll
        for (unsigned sb = 0; sb < store_block_batch_size; sb++) {
#pragma unroll
            for (unsigned j = 0;
                 j < size_of<BLOCK_T>::value / size_of<half2>::value; j++) {
                batch_block[sb].u[j] = curand(&curand_gen);
                batch_block[sb].u[j] |= 0x00010001u;
            }
        }
#pragma unroll
        for (unsigned sb = 0; sb < store_block_batch_size; sb++) {
#pragma unroll
            for (unsigned j = 0;
                 j < size_of<BLOCK_T>::value / size_of<uint1>::value; j++) {
                const auto s2 = batch_block[sb].s2[j];
                const auto f2_x = static_cast<float>(s2.x) / 0x10000;
                const auto mag = sigma * sqrtf(-2.0f * logf(f2_x));
                constexpr float two_pi = M_PI * 2;
                const auto f2_y = static_cast<float>(s2.y) / 0x10000;
                float si, co;
                sincosf(two_pi * f2_y, &si, &co);
                const auto h2 = __float22half2_rn(make_float2(si, co));
                batch_block[sb].h2[j] =
                    __hfma2(__float2half2_rn(mag), h2, __float2half2_rn(mean));
            }
        }
#pragma unroll
        for (unsigned sb = 0; sb < store_block_batch_size; sb++) {
            *(reinterpret_cast<BLOCK_T*>(array_ptr + i) + sb) =
                batch_block[sb].store_block;
        }
    }
    if (tid == 0) {
        const auto res = size - batch_loop_size;
        if (res != 0) {
            for (unsigned j = 0; j < res; j++) {
                const auto v = (curand_normal(&curand_gen) + mean) * sigma;
                array_ptr[batch_loop_size + j] = __float2half(v);
            }
        }
    }
    *(status_ptr + tid) = curand_gen;
}
}  // namespace

template <curandRngType_t rng>
struct curand_status_t {
    using type = void;
};

template <>
struct curand_status_t<CURAND_RNG_PSEUDO_MTGP32> {
    using type = curandStateMtgp32_t;
};

template <>
struct curand_status_t<CURAND_RNG_QUASI_SCRAMBLED_SOBOL32> {
    using type = curandStateScrambledSobol32_t;
};

template <>
struct curand_status_t<CURAND_RNG_QUASI_SOBOL32> {
    using type = curandStateSobol32_t;
};

template <>
struct curand_status_t<CURAND_RNG_PSEUDO_MRG32K3A> {
    using type = curandStateMRG32k3a_t;
};

template <>
struct curand_status_t<CURAND_RNG_PSEUDO_XORWOW> {
    using type = curandStateXORWOW_t;
};

template <>
struct curand_status_t<CURAND_RNG_PSEUDO_PHILOX4_32_10> {
    using type = curandStatePhilox4_32_10_t;
};

struct generator_t {
    unsigned num_sm;
    unsigned num_threads;
    cudaStream_t cuda_stream;
    curandRngType_t rng_type;
    void* status_ptr;
};

void Create(generator_t& gen, const curandRngType_t rng_type) {
    // set cuda stream
    gen.cuda_stream = 0;
    // get num sm
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    gen.num_sm = prop.multiProcessorCount;

    // calculate grid_size
    gen.num_threads = gen.num_sm * num_sm_scale * block_size;

    // set algo
    gen.rng_type = rng_type;

    // set generator
    unsigned state_struct_size = 0;
    switch (rng_type) {
#define CASE_RNG_TYPE(rng)                                               \
    case rng:                                                            \
        state_struct_size = sizeof(typename curand_status_t<rng>::type); \
        break
        CASE_RNG_TYPE(CURAND_RNG_PSEUDO_MRG32K3A);
        CASE_RNG_TYPE(CURAND_RNG_PSEUDO_XORWOW);
        CASE_RNG_TYPE(CURAND_RNG_PSEUDO_PHILOX4_32_10);
        default:
            throw std::runtime_error("Unknown pseudo rand algorithm");
#undef CASE_RNG_TYPE
    }
    const auto stat =
        cudaMalloc(&gen.status_ptr, state_struct_size * gen.num_threads);
    if (stat != cudaSuccess) {
        throw std::runtime_error(
            "[curand_fp16 error] : " + std::string(cudaGetErrorString(stat)) +
            " @" + __func__);
    }
};

void Destroy(generator_t& gen) { cudaFree(gen.status_ptr); };

void SetSeed(generator_t& gen, const std::uint64_t seed) {
    switch (gen.rng_type) {
#define CASE_RNG_TYPE(rng)                                              \
    case rng:                                                           \
        StatusInitKernel<typename curand_status_t<rng>::type>           \
            <<<gen.num_threads / block_size, block_size, 0,             \
               gen.cuda_stream>>>(                                      \
                reinterpret_cast<typename curand_status_t<rng>::type*>( \
                    gen.status_ptr),                                    \
                seed);                                                  \
        break
        CASE_RNG_TYPE(CURAND_RNG_PSEUDO_MRG32K3A);
        CASE_RNG_TYPE(CURAND_RNG_PSEUDO_XORWOW);
        CASE_RNG_TYPE(CURAND_RNG_PSEUDO_PHILOX4_32_10);
        default:
            throw std::runtime_error("Unknown pseudo rand algorithm");
#undef CASE_RNG_TYPE
    }
};

void SetCudaStream(generator_t& generator, cudaStream_t const cuda_stream) {
    generator.cuda_stream = cuda_stream;
};

// Uniform distribution
// pm == true  | (-1, 1)
//       false | ( 0, 1)
void Uniform(generator_t& gen, half* const ptr, const std::size_t size,
             const bool pm = false) {
    const auto batch_size = size_of<block_t>::value / size_of<half>::value;
    const auto grid_size = std::min<unsigned>(
        std::min<unsigned>(gen.num_threads / block_size,
                           (size + block_size - 1) / block_size),
        ((size + batch_size - 1) / batch_size + block_size - 1) / block_size);
    if (pm == 0) {
        switch (gen.rng_type) {
#define CASE_RNG_TYPE(rng)                                              \
    case rng:                                                           \
        GenerateKernel<typename curand_status_t<rng>::type, block_t, 0> \
            <<<grid_size, block_size, 0, gen.cuda_stream>>>(            \
                ptr,                                                    \
                reinterpret_cast<typename curand_status_t<rng>::type*>( \
                    gen.status_ptr),                                    \
                size);                                                  \
        break
            CASE_RNG_TYPE(CURAND_RNG_PSEUDO_MRG32K3A);
            CASE_RNG_TYPE(CURAND_RNG_PSEUDO_XORWOW);
            CASE_RNG_TYPE(CURAND_RNG_PSEUDO_PHILOX4_32_10);
            default:
                throw std::runtime_error("Unknown pseudo rand algorithm");
#undef CASE_RNG_TYPE
        }
    } else {
        switch (gen.rng_type) {
#define CASE_RNG_TYPE(rng)                                              \
    case rng:                                                           \
        GenerateKernel<typename curand_status_t<rng>::type, block_t, 1> \
            <<<grid_size, block_size, 0, gen.cuda_stream>>>(            \
                ptr,                                                    \
                reinterpret_cast<typename curand_status_t<rng>::type*>( \
                    gen.status_ptr),                                    \
                size);                                                  \
        break
            CASE_RNG_TYPE(CURAND_RNG_PSEUDO_MRG32K3A);
            CASE_RNG_TYPE(CURAND_RNG_PSEUDO_XORWOW);
            CASE_RNG_TYPE(CURAND_RNG_PSEUDO_PHILOX4_32_10);
            default:
                throw std::runtime_error("Unknown pseudo rand algorithm");
#undef CASE_RNG_TYPE
        }
    }
};

// Normal distribution
void Normal(generator_t& gen, half* const ptr, const std::size_t size,
            const float mean, const float var) {
    const auto batch_size = size_of<block_t>::value / size_of<half>::value;
    const auto grid_size = std::min<unsigned>(
        std::min<unsigned>(gen.num_threads / block_size,
                           (size + block_size - 1) / block_size),
        ((size + batch_size - 1) / batch_size + block_size - 1) / block_size);
    switch (gen.rng_type) {
#define CASE_RNG_TYPE(rng)                                                 \
    case rng:                                                              \
        GenerateNormalKernel<typename curand_status_t<rng>::type, block_t> \
            <<<grid_size, block_size, 0, gen.cuda_stream>>>(               \
                ptr,                                                       \
                reinterpret_cast<typename curand_status_t<rng>::type*>(    \
                    gen.status_ptr),                                       \
                size, mean, std::sqrt(var));                               \
        break
        CASE_RNG_TYPE(CURAND_RNG_PSEUDO_MRG32K3A);
        CASE_RNG_TYPE(CURAND_RNG_PSEUDO_XORWOW);
        CASE_RNG_TYPE(CURAND_RNG_PSEUDO_PHILOX4_32_10);
        default:
            throw std::runtime_error("Unknown pseudo rand algorithm");
#undef CASE_RNG_TYPE
    }
};
}  // namespace curand_fp16

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
