#pragma once

#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/gemm_kernel_traits.h"
#include "kaleido/core/device/kernels/gmem_tile_transmitter.h"
#include "kaleido/core/device/kernels/smem_tile_transmitter.h"

#include <cublas_v2.h>
#include <cutlass/numeric_types.h>
#include <glog/logging.h>

#include <algorithm>

namespace kaleido {
namespace core {
namespace cuda_kernel {

namespace {
constexpr int GetGcd(int m, int n) {
    int z = n;
    while (m % n != 0) {
        z = m % n;
        m = n;
        n = z;
    }
    return z;
}
}  // namespace

template <typename T>
struct CublasGemm {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const T* alpha, const T* A, int lda, const T* B, int ldb,
                    const T* beta, T* C, int ldc);
};

template <>
struct CublasGemm<float> {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const float* alpha, const float* A, int lda, const float* B,
                    int ldb, const float* beta, float* C, int ldc) {
        CublasCheck(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda,
                                B, ldb, beta, C, ldc));
    }
};

template <>
struct CublasGemm<__half> {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const __half* alpha, const __half* A, int lda,
                    const __half* B, int ldb, const __half* beta, __half* C,
                    int ldc) {
        CublasCheck(cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda,
                                B, ldb, beta, C, ldc));
    }
};

template <>
struct CublasGemm<cutlass::half_t> {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const cutlass::half_t* alpha, const cutlass::half_t* A,
                    int lda, const cutlass::half_t* B, int ldb,
                    const cutlass::half_t* beta, cutlass::half_t* C, int ldc) {
        CublasCheck(cublasHgemm(handle, transa, transb, m, n, k,
                                reinterpret_cast<const __half*>(alpha),
                                reinterpret_cast<const __half*>(A), lda,
                                reinterpret_cast<const __half*>(B), ldb,
                                reinterpret_cast<const __half*>(beta),
                                reinterpret_cast<__half*>(C), ldc));
    }
};

/// host API for cutlass wmma.
template <typename Element, /*dtype*/
          typename InstructionShape, typename WarpShape,
          typename ThreadBlockShape, typename WholeShape>
struct CutlassWarpGemm {
    void operator()(const Element* A, const Element* B, Element* C) {
        // Whole GEMM shape
        const int kM = WholeShape::kM;
        const int kN = WholeShape::kN;
        const int kK = WholeShape::kK;

        // CTA GEMM shape
        const int kTM = ThreadBlockShape::kM;
        const int kTN = ThreadBlockShape::kN;
        const int kTK = ThreadBlockShape::kK;

        const int kNumMmaWarp = kTM / WarpShape::kM * kTN / WarpShape::kN;
        const int kThreads = kNumMmaWarp * 32;

        const int kNumElementPerAccess =
            128 / cutlass::sizeof_bits<Element>::value;
        // 128B is the shared memory cache line width
        const int crosswise =
            128 * 8 / cutlass::sizeof_bits<Element>::value / 2;
        using SLayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
            cutlass::sizeof_bits<Element>::value, crosswise /*crosswise*/>;
        using SLayoutB =
            cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
                cutlass::sizeof_bits<Element>::value, crosswise /*crosswise*/>;
        using Mma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
            WarpShape, InstructionShape, Element, SLayoutA, Element, SLayoutB,
            Element, cutlass::layout::RowMajor>::Type;

        const int kNumWarpA = kTM * kTK / kNumElementPerAccess / 32;
        const int kThreadsA = GetGcd(kNumWarpA, kNumMmaWarp) * 32;
        using LoadA = GmemTileTransmitter<kTM, kTK, Element, kThreadsA,
                                          TileLayout::RowMajor,
                                          TileLayout::SwizzledRowMajor>;
        const int kNumWarpB = kTN * kTK / kNumElementPerAccess / 32;
        const int kThreadsB = GetGcd(kNumWarpB, kNumMmaWarp) * 32;
        /*NOTE: interperate B is laid out in column-major in the
         * global memory.*/
        using LoadB = GmemTileTransmitter<kTK, kTN, Element, kThreadsB,
                                          TileLayout::ColumnMajor,
                                          TileLayout::SwizzledColumnMajor>;
        const int kNumWarpC = kTM * kTN / kNumElementPerAccess / 32;
        const int kThreadsC = GetGcd(kNumWarpC, kNumMmaWarp) * 32;
        using StoreC =
            SmemTileTransmitter<kTM, kTN, Element, kThreadsC,
                                TileLayout::RowMajor, TileLayout::RowMajor>;
        LoadA loader_a(kTM, kTK);
        LoadB loader_b(kTK, kTN);
        StoreC storer_c(kTM, kTN);

        const int block_m = CeilDiv<kM, kTM>;
        const int block_n = CeilDiv<kN, kTN>;

        dim3 gridDim(block_m, block_n);
        dim3 blockDim(kThreads, 1, 1);

        const int smem_count =
            kTM * kTN <= kTK * (kTN + kTM) ? kTK * (kTN + kTM) : kTM * kTN;

        KeGemm<Element, Mma, WholeShape, ThreadBlockShape, WarpShape,
               InstructionShape, LoadA, LoadB, StoreC, smem_count>
            <<<gridDim, blockDim>>>(A, B, C, loader_a, loader_b, storer_c);
    }
};

/* NOTE: In the current implementation, operands A, B and C are
fixed to have the following layouts: A: RowMajor, B: ColumnMajor, C:
RowMajor
 */
template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteGemm {
    void operator()(const Element* A, const Element* B, Element* C) {
        // Whole GEMM shape
        static const int kM = dim_size<0, WholeShape>;
        static const int kN = dim_size<1, WholeShape>;
        static const int kK = dim_size<2, WholeShape>;

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
            KeGemmTraits<Element, InstructionShape, ValueMnk, WarpArrangement,
                         CtaTileShape, WholeShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM), kTM * kTN) * sizeof(Element);

        auto kernel = &KeCuteGemm<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }

        const int block_m = CeilDiv<kM, kTM>;
        const int block_n = CeilDiv<kN, kTN>;

        const int kThreads = KeTraits::kThreads;

        dim3 gridDim(block_m, block_n);
        dim3 blockDim(kThreads, 1, 1);
        kernel<<<gridDim, blockDim, smem_size>>>(A, B, C);
    }
};

template <typename Element, const int num_stages, const int kThreads,
          const int kWarpPerRow, typename CtaTileShape, typename WholeShape>
struct GemmPipelined {
    void operator()(const Element* dA, const Element* dB, Element* dC) {
        // Whole GEMM shape
        const int kM = dim_size<0, WholeShape>;
        const int kN = dim_size<1, WholeShape>;
        const int kK = dim_size<2, WholeShape>;

        // CTA GEMM shape
        const int kTM = dim_size<0, CtaTileShape>;
        const int kTN = dim_size<1, CtaTileShape>;
        const int kTK = dim_size<2, CtaTileShape>;

        int shm_AB = sizeof(Element) * (kTM + kTN) * kTK * num_stages;
        int shm_C = sizeof(Element) * (kTM * kTN);
        int shm = shm_AB < shm_C ? shm_C : shm_AB;

        using KeTraits =
            KePipelinedGemmTraits<Element, num_stages, kThreads, kWarpPerRow,
                                  CtaTileShape, WholeShape>;
        auto kernel = &KeGemmPipelined<Element, KeTraits>;

        if (shm > 48 * 1024) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
        }

        int blocks = CeilDiv<kM * kN, kTM * kTM>;
        kernel<<<dim3(blocks, 1, 1), dim3(kThreads, 1, 1), shm, 0>>>(dA, dB,
                                                                     dC);
    }
};

// host API to launch the kernel for back-to-back GEMM.
template <typename Element, typename WholeShape, typename CtaTileShape,
          typename WarpShape>
struct B2BGemm {
    void operator()(const Element* dA, const Element* dB, const Element* dC,
                    Element* dD) {
        static const int kM = dim_size<0, WholeShape>;
        static const int kN = dim_size<1, WholeShape>;
        static const int kK = dim_size<2, WholeShape>;
        static const int kP = dim_size<3, WholeShape>;

        static const int kTM = dim_size<0, CtaTileShape>;
        static const int kTN = dim_size<1, CtaTileShape>;
        static const int kTK = dim_size<2, CtaTileShape>;
        static const int kTP = dim_size<3, CtaTileShape>;

        using KeTraits =
            KeBack2BackGemmTraits<Element, WholeShape, CtaTileShape, WarpShape>;

        int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
        int shm_output = kTM * kTP;
        int shm_size = shm_input < shm_output ? shm_output * sizeof(Element)
                                              : shm_input * sizeof(Element);

        auto kernel = &KeBack2BackGemm<Element, KeTraits>;
        if (shm_size > 48 * 1024) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
        }

        // blocks are launched along the M and P dimensions.
        int block_x = CeilDiv<kM, kTM>;
        int block_y = CeilDiv<kP, kTP>;

        kernel<<<dim3(block_x, block_y, 1), dim3(KeTraits::kThreads, 1, 1),
                 shm_size, 0>>>(dA, dB, dC, dD);
    }
};

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
