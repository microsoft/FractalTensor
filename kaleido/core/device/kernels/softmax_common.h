#pragma once

namespace kaleido {
namespace core {
namespace cuda_kernel {

template <typename T>
struct MD {
    T m;
    T d;
};

template <>
struct __align__(8) MD<float> {
    float m;
    float d;
};

template <>
struct __align__(16) MD<double> {
    double m;
    double d;
};

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
