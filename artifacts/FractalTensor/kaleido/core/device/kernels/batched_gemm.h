#pragma once
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/gemm_kernel_traits.h"

#include <cublas_v2.h>
#include <cutlass/numeric_types.h>

namespace kaleido {
namespace core {
namespace cuda_kernel {

template <typename T>
struct CublasGemmBatched {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const T* alpha, const T** Aarray, int lda, const T** Barray,
                    int ldb, const T* beta, T** Carray, int ldc,
                    int batchCount) {}
};

template <>
struct CublasGemmBatched<float> {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const float* alpha, const float** Aarray, int lda,
                    const float** Barray, int ldb, const float* beta,
                    float** Carray, int ldc, int batchCount) {
        CublasCheck(cublasSgemmBatched(handle, transa, transb, m, n, k, alpha,
                                       Aarray, lda, Barray, ldb, beta, Carray,
                                       ldc, batchCount));
    }
};

template <>
struct CublasGemmBatched<cutlass::half_t> {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const cutlass::half_t* alpha, const cutlass::half_t** A,
                    int lda, const cutlass::half_t** B, int ldb,
                    const cutlass::half_t* beta, cutlass::half_t** C, int ldc,
                    int batchCount) {
        CublasCheck(cublasHgemmBatched(handle, transa, transb, m, n, k,
                                       reinterpret_cast<const __half*>(alpha),
                                       reinterpret_cast<const __half**>(A), lda,
                                       reinterpret_cast<const __half**>(B), ldb,
                                       reinterpret_cast<const __half*>(beta),
                                       reinterpret_cast<__half**>(C), ldc,
                                       batchCount));
    }
};

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteBatchedGemm {
    void operator()(const Element* A, const Element* B, Element* C) {
        // Whole GEMM shape
        static const int kM = dim_size<0, WholeShape>;
        static const int kN = dim_size<1, WholeShape>;
        static const int kK = dim_size<2, WholeShape>;
        static const int kB = dim_size<3, WholeShape>;

        // CTA GEMM shape
        static const int kTM = dim_size<0, CtaTileShape>;
        static const int kTN = dim_size<1, CtaTileShape>;
        static const int kTK = dim_size<2, CtaTileShape>;

        static_assert(kTM % dim_size<0, WarpArrangement> == 0,
                      "the M dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");
        static_assert(kTN % dim_size<1, WarpArrangement> == 0,
                      "the N dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");

        using KeTraits =
            KeBatchedGemmTraits<Element, InstructionShape, ValueMnk,
                                WarpArrangement, CtaTileShape, WholeShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM), kTM * kTN) * sizeof(Element);

        auto kernel = &KeBatchedCuteGemm<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }

        const int block_m = CeilDiv<kM, kTM>;
        const int block_n = CeilDiv<kN, kTN>;

        const int kThreads = KeTraits::kThreads;

        // TODO: check if the grid size is too large
        dim3 gridDim(block_m, block_n, kB);
        dim3 blockDim(kThreads, 1, 1);
        kernel<<<gridDim, blockDim, smem_size>>>(A, B, C);
    }
};

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
