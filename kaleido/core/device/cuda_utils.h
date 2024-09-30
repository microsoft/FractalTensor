#pragma once

#include "kaleido/core/config.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

namespace kaleido {
namespace core {

template <int a, int b>
constexpr int CeilDiv = (a + b - 1) / b;  // for compile-time values

#define DIVUP(a, b) (a) + (b) - (1) / (b)  // for runtime values

namespace {
const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown error";
}

const char* curandGetErrorString(curandStatus_t status) {
    switch (status) {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "Unknown cuRAND error";
}

}  // namespace

inline void __cudaCheck(const cudaError err, const char* file, int line) {
#ifndef NDEBUG
    if (err != cudaSuccess) {
        fprintf(stderr, "%s(%d): CUDA error: %s.\n", file, line,
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif
}

inline void __cublasCheck(const cublasStatus_t err, const char* file,
                          int line) {
#ifndef NDEBUG
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s(%d): Cublas error: %s.\n", file, line,
                cublasGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif
}

inline void __cudnnCheck(const cudnnStatus_t err, const char* file, int line) {
#ifndef NDEBUG
    if (err != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "%s(%d): CuDNN error: %s.\n", file, line,
                cudnnGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif
}

inline void __curandCheck(curandStatus_t err, const char* file, int line) {
#ifndef NDEBUG
    if (err != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "%s(%d): Curand error: %s.\n", file, line,
                curandGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif
}

#define CudaCheck(call) __cudaCheck(call, __FILE__, __LINE__)
#define CublasCheck(call) __cublasCheck(call, __FILE__, __LINE__)
#define CudnnCheck(call) __cudnnCheck(call, __FILE__, __LINE__)
#define CurandCheck(call) __curandCheck(call, __FILE__, __LINE__)
}  // namespace core
}  // namespace kaleido
