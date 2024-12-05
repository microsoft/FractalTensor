#pragma once

#if defined(__CUDA_ARCH__)
#define HOST_DEVICE __forceinline__ __host__ __device__
#define DEVICE __forceinline__ __device__
#define HOST __forceinline__ __host__
#else
#define HOST_DEVICE inline
#define DEVICE inline
#define HOST inline
#endif

#if defined(__CUDACC_RTC__)
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#define STL_NAMESPACE cuda::std

#else
#include <cstddef>  // ptrdiff_t
#include <cstdint>  // uintptr_t
#include <limits>   // numeric_limits
#include <type_traits>
#include <utility>  // tuple_size, tuple_element

#define STL_NAMESPACE std
#endif
