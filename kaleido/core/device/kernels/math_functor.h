#pragma once
#include <math.h>

namespace kaleido {
namespace core {
namespace cuda_kernel {

template <typename T>
struct Exp {
    __forceinline__ __host__ __device__ T operator()(const T& a) const;
};

template <>
struct Exp<float> {
    __forceinline__ __host__ __device__ float operator()(const float& a) const {
#ifdef USE_FAST_MATH
        return __expf(a);
#else
        return exp(a);
#endif
    }
};

template <>
struct Exp<double> {
    __forceinline__ __host__ __device__ double operator()(
        const double& a) const {
        return exp(a);
    }
};

template <typename T>
struct Log {
    __forceinline__ __host__ __device__ T operator()(const T& a) const;
};

template <>
struct Log<float> {
    __forceinline__ __host__ __device__ float operator()(const float& a) const {
#ifdef USE_FAST_MATH
        return __logf(a);
#else
        return log(a);
#endif
    }
};

template <>
struct Log<double> {
    __forceinline__ __host__ __device__ double operator()(
        const double& a) const {
        return log(a);
    }
};

template <typename T>
struct Add {
    __forceinline__ __host__ __device__ T operator()(const T& a,
                                                     const T& b) const {
        return a + b;
    }
};

template <typename T>
struct Max {
    __forceinline__ __host__ __device__ T operator()(const T& a,
                                                     const T& b) const {
        return a > b ? a : b;
    }
};

template <typename T>
struct Min {
    __forceinline__ __host__ __device__ T operator()(const T& a,
                                                     const T& b) const {
        return a > b ? b : a;
    }
};

template <typename T>
struct Prod {
    __forceinline__ __host__ __device__ T operator()(const T& a,
                                                     const T& b) const {
        return a * b;
    }
};

template <typename T>
struct Sub {
    __forceinline__ __host__ __device__ T operator()(const T a,
                                                     const T b) const {
        return a - b;
    }
};

template <typename T>
struct Multiply {
    __forceinline__ __host__ __device__ T operator()(const T a,
                                                     const T b) const {
        return a * b;
    }
};

template <>
struct Multiply<bool> {
    __forceinline__ __host__ __device__ bool operator()(const bool a,
                                                        const bool b) const {
        return a && b;
    }
};

template <typename T, typename Enable = void>
struct Div {
    __forceinline__ __host__ __device__ T operator()(const T a,
                                                     const T b) const;
};

template <>
struct Div<double> {
    __forceinline__ __host__ __device__ double operator()(
        const double a, const double b) const {
#ifdef USE_FAST_MATH
        return __fdividef(a, b);
#else
        return a / b;
#endif
    }
};

template <>
struct Div<float> {
    __forceinline__ __host__ __device__ double operator()(const float a,
                                                          const float b) const {
        return a / b;
    }
};

template <typename T>
struct Div<T, typename std::enable_if<std::is_integral<T>::value>::type> {
    __forceinline__ __host__ __device__ T operator()(const T a,
                                                     const T b) const {
        // For int32/int64, need to check whether the divison is
        // zero.
        CHECK(b != 0);
        return a / b;
    }
};

template <typename T, typename Enable = void>
struct InverseDiv {
    __forceinline__ __host__ __device__ T operator()(const T a,
                                                     const T b) const {
        return b / a;
    }
};

template <typename T>
struct AddAndTanh {  // for simulate elementwise fusion.
    __forceinline__ __host__ __device__ T operator()(const T& a,
                                                     const T& b) const {
        return tanh(a + b);
    }
};

template <typename T>
struct Sigmoid {
    __forceinline__ __host__ __device__ T operator()(const T& a) const {
        return 1. / (1. + exp(-a));
    }
};

template <typename T>
struct Tanh {
    __forceinline__ __host__ __device__ T operator()(const T& a) const {
        return tanh(a);
    }
};

template <typename T>
struct CellFunc {
    __forceinline__ __host__ __device__ T
    operator()(const T& f, const T& c, const T& i, const T& c_candidate) const {
        return f * c + i * c_candidate;
    }
};

template <typename T>
struct HiddenFunc {
    __forceinline__ __host__ __device__ T operator()(const T& o,
                                                     const T& c) const {
        return o * tanh(c);
    }
};

template <typename T>
struct SubAndExp {
    __forceinline__ __device__ __host__ SubAndExp(T scale) : scale(scale) {}

    __forceinline__ __host__ __device__ T operator()(const T& a) const {
        return exp(a - scale);
    }

    T scale;
};

template <typename T>
struct Indentity {
    __forceinline__ __host__ __device__ T operator()(const T& a) const {
        return a;
    }
};

template <typename T>
struct Inverse {
    __forceinline__ __host__ __device__ T operator()(const T& a) const {
        return static_cast<T>(1) / a;
    }
};

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
