#pragma once

#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/print_op.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cutlass/numeric_types.h>
#include <gtest/gtest.h>

#include <iomanip>

namespace kaleido {
namespace core {
namespace test_utils {

template <typename T>
std::shared_ptr<Tensor> SequentialGpuTensor(
    std::initializer_list<int64_t> sizes, std::shared_ptr<Allocator> alloc) {
    auto x = std::make_shared<Tensor>(sizes, alloc);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    ops::FillOp<GPUContext, CUDAPlace, T> f;
    f(*x, "seq");

    return x;
}

template <typename T>
std::shared_ptr<Tensor> ConstantGpuTensor(std::initializer_list<int64_t> sizes,
                                          std::shared_ptr<Allocator> alloc,
                                          float val) {
    auto x = std::make_shared<Tensor>(sizes, alloc);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    ops::FillOp<GPUContext, CUDAPlace, T> f;
    f(*x, val);

    return x;
}

template <typename T>
std::shared_ptr<Tensor> RandomGpuTensor(std::initializer_list<int64_t> sizes,
                                        std::shared_ptr<Allocator> alloc,
                                        T mean = 0, T stddev = 0.1) {
    auto x = std::make_shared<Tensor>(sizes, alloc);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    ops::FillOp<GPUContext, CUDAPlace, T> frand;
    frand(*x);

    return x;
}

template <typename T>
void PrintTensor(const Tensor& data, int precision = 3) {
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    ops::PrintOp<GPUContext, CUDAPlace, T> fprint;
    std::cout << fprint(data, precision);
}

// offset in column major storage.
// M = num_rows, N = num_cols
#define offset_cm(i, j, M, N) (i + j * M)
#define offset_rm(i, j, M, N) (i * N + j)

void CompareWithNaiveMatMulColMajor(const float* dA, bool transa,
                                    const float* dB, bool transb,
                                    const float* dC, int64_t m, int64_t n,
                                    int64_t k) {
    /* This naive implementation is for debugging.
     * A, B and C are matrices that are stored in column-major.
     * A: [m, k], B: [k, n], C: [m, n]
     */
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double tmp = 0.;
            for (size_t p = 0; p < k; ++p) {
                if (transa == false and transb == false) {
                    tmp +=
                        dA[offset_cm(i, p, m, k)] * dB[offset_cm(p, j, k, n)];
                } else if (transa == true and transb == false) {
                    tmp +=
                        dA[offset_cm(p, i, k, m)] * dB[offset_cm(p, j, k, n)];
                } else if (transa == false and transb == true) {
                    tmp +=
                        dA[offset_cm(i, p, m, k)] * dB[offset_cm(j, p, n, k)];
                } else if (transa == true and transb == true) {
                    tmp +=
                        dA[offset_cm(p, i, k, m)] * dB[offset_cm(j, p, n, k)];
                }
            }
            EXPECT_FLOAT_EQ(tmp, dC[offset_cm(i, j, m, n)]);
        }
    }
}

void CompareWithNaiveMatMulRowMajor(const float* dA, bool transa,
                                    const float* dB, bool transb,
                                    const float* dC, int64_t m, int64_t n,
                                    int64_t k) {
    /* This naive implementation is for debugging.
     * A, B and C are matrices that are stored in column-major.
     * A: [m, k], B: [k, n], C: [m, n]
     */
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double tmp = 0.;
            for (size_t p = 0; p < k; ++p) {
                if (transa == false and transb == false) {
                    tmp +=
                        dA[offset_rm(i, p, m, k)] * dB[offset_rm(p, j, k, n)];
                } else if (transa == true and transb == false) {
                    tmp +=
                        dA[offset_rm(p, i, k, m)] * dB[offset_rm(p, j, k, n)];
                } else if (transa == false and transb == true) {
                    tmp +=
                        dA[offset_rm(i, p, m, k)] * dB[offset_rm(j, p, n, k)];
                } else if (transa == true and transb == true) {
                    tmp +=
                        dA[offset_rm(p, i, k, m)] * dB[offset_rm(j, p, n, k)];
                }
            }
            EXPECT_FLOAT_EQ(tmp, dC[offset_rm(i, j, m, n)]);
        }
    }
}

template <typename T>
__device__ void DebugPrint(const T* data, int row, int col) {
    printf("\nDump all the elements:\n");

    for (int i = 0; i < row; ++i) {
        printf("%d\t", i);
        for (int j = 0; j < col; ++j) {
            printf("%.0f ", T(data[i * col + j]));
        }
        printf("\n");
    }
}

template <>
__device__ void DebugPrint(const cutlass::half_t* data, int row, int col) {
    printf("\nDump all the elements:\n");

    for (int i = 0; i < row; ++i) {
        printf("%d\t", i);
        for (int j = 0; j < col; ++j) {
            printf("%.3f ", float(data[i * col + j] * 2.0_hf));
        }
        printf("\n");
    }
}

}  // namespace test_utils
}  // namespace core
}  // namespace kaleido
