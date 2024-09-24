#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/batched_gemm.h"
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/gemm.h"
#include "kaleido/core/operators/expect_eq_op.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/print_op.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tile_shape.h"

#include <cutlass/gemm/device/gemm.h>
#include <gtest/gtest.h>

namespace kaleido {
namespace core {

template <typename Element, /*dtype*/
          typename InstructionShape, typename WarpShape,
          typename ThreadBlockShape, typename WholeShape>
void run_cutlass2_gemm() {
    // GEMM shape
    int kM = WholeShape::kM;
    int kN = WholeShape::kN;
    int kK = WholeShape::kK;

    // CTA shape
    const int kTM = ThreadBlockShape::kM;
    const int kTN = ThreadBlockShape::kN;
    const int kTK = ThreadBlockShape::kK;

    const int kNumMmaWarp = kTM / WarpShape::kM * kTN / WarpShape::kN;
    const int kThreads = kNumMmaWarp * 32;

    /// ==== Initialize Test Data ====
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    Tensor A({kM, kK}, allocator);
    Tensor B({kK, kN}, allocator);
    Tensor C({kM, kN}, allocator);
    Tensor ref_C({kM, kN}, allocator);

    fill(A, 0, 1e-3);
    fill(B, 0, 1e-3);

    fill(C, 0.);
    fill(ref_C, 0.);

    /// cutlass GEMM
    cuda_kernel::CutlassWarpGemm<Element, InstructionShape, WarpShape,
                                 ThreadBlockShape, WholeShape>
        gemm;

    /// cuBLAS gemm as the groundtruth
    cuda_kernel::CublasGemm<Element> gemm_reference;
    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));
    Element alf = static_cast<Element>(1.);
    Element bet = static_cast<Element>(0.);

    // unittest
    gemm(A.data<Element>(), B.data<Element>(), C.mutable_data<Element>());
    gemm_reference(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kN, kM, kK,
                   &alf, B.data<Element>(), B.dim_size(0), A.data<Element>(),
                   A.dim_size(1), &bet, ref_C.mutable_data<Element>(),
                   ref_C.dim_size(1));

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(C, ref_C, 1e-3);

    CublasCheck(cublasDestroy(handle));
}

/*
 * The cute_gemm function performs matrix multiplication using the warp-level
 * matrix multiply-accumulate (wmma) instructions. The performance and
 * resource usage of the function can be controlled by adjusting the values of
 * WarpScale and CTAScale. Increasing WarpScale will cause each thread to
 * occupy more registers, resulting in more frequent execution of wmma
 * instructions. Increasing CTAScale will launch more warps to execute the
 * program, using a larger CTA tile and requiring more shared memory.
 */
template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_gemm() {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    cudaDeviceProp m_dev_prop;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&m_dev_prop, device_idx);

    // GEMM shape
    int kM = dim_size<0, WholeShape>;
    int kN = dim_size<1, WholeShape>;
    int kK = dim_size<2, WholeShape>;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor A({kM, kK}, allocator);
    kaleido::core::Tensor B({kK, kN}, allocator);
    kaleido::core::Tensor C({kM, kN}, allocator);

    kaleido::core::Tensor ref_C({kM, kN}, allocator);

    fill(A, 0, 1e-3);
    fill(B, 0, 1e-3);
#ifdef DEBUG
    fill(A, "seq");
    fill(B, "seq");
#endif

    fill(C, 0.);
    fill(ref_C, 0.);

    using GEMM =
        cuda_kernel::CuteGemm<Element, InstructionShape, ValueMnk,
                              WarpArrangement, CtaTileShape, WholeShape>;
    GEMM gemm;
    gemm(A.data<Element>(), B.data<Element>(), C.mutable_data<Element>());

    // cuBLAS gemm as the groundtruth
    cuda_kernel::CublasGemm<Element> gemm_reference;
    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));
    Element alf = static_cast<Element>(1.);
    Element bet = static_cast<Element>(0.);

    gemm_reference(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kN, kM, kK,
                   &alf, B.data<Element>(), B.dim_size(0), A.data<Element>(),
                   A.dim_size(1), &bet, ref_C.mutable_data<Element>(),
                   ref_C.dim_size(1));

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(C, ref_C, 5e-3);

    std::cout << "Passed unittest for [" << kM << ", " << kN << ", " << kK
              << "]" << std::endl;
}

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_batched_cute_gemm() {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    cudaDeviceProp m_dev_prop;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&m_dev_prop, device_idx);

    // GEMM shape
    int kM = dim_size<0, WholeShape>;
    int kN = dim_size<1, WholeShape>;
    int kK = dim_size<2, WholeShape>;
    int kB = dim_size<3, WholeShape>;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor A({kB, kM, kK}, allocator);
    kaleido::core::Tensor B({kB, kK, kN}, allocator);
    kaleido::core::Tensor C({kB, kM, kN}, allocator);

    kaleido::core::Tensor ref_C({kB, kM, kN}, allocator);

    fill(A, 0, 1e-3);
    fill(B, 0, 1e-3);
#ifdef DEBUG
    fill(A, "seq");
    fill(B, "seq");
#endif

    fill(C, 0.);
    fill(ref_C, 0.);

    using BatchedGEMM =
        cuda_kernel::CuteBatchedGemm<Element, InstructionShape, ValueMnk,
                                     WarpArrangement, CtaTileShape, WholeShape>;
    BatchedGEMM batched_gemm;
    batched_gemm(A.data<Element>(), B.data<Element>(),
                 C.mutable_data<Element>());

    std::cout << "A.dim_size: " << A.dim_size(0) << ", " << A.dim_size(1)
              << ", " << A.dim_size(2) << std::endl;
    std::cout << "B.dim_size: " << B.dim_size(0) << ", " << B.dim_size(1)
              << ", " << B.dim_size(2) << std::endl;
    std::cout << "C.dim_size: " << C.dim_size(0) << ", " << C.dim_size(1)
              << ", " << C.dim_size(2) << std::endl;

    cuda_kernel::CublasGemm<Element> gemm_reference;
    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));
    Element alf = static_cast<Element>(1.);
    Element bet = static_cast<Element>(0.);

    for (int i = 0; i < kB; i++) {
        gemm_reference(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kN, kM, kK,
                       &alf, B.data<Element>() + i * kN * kK, B.dim_size(1),
                       A.data<Element>() + i * kM * kK, A.dim_size(2), &bet,
                       ref_C.mutable_data<Element>() + i * kM * kN,
                       ref_C.dim_size(2));
    }

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(C, ref_C, 5e-3);

    std::cout << "Passed unittest for [" << kM << ", " << kN << ", " << kK
              << ", " << kB << "]" << std::endl;
}

TEST(TestCutlassGEMM, test1) {
    run_cutlass2_gemm<cutlass::half_t,
                      cutlass::gemm::GemmShape<16, 8, 8> /*instruction shape*/,
                      cutlass::gemm::GemmShape<64, 32, 32> /*warp tile shape*/,
                      cutlass::gemm::GemmShape<64, 64, 64> /*CTA tile shape*/,
                      cutlass::gemm::GemmShape<128, 128, 128>>();

    run_cutlass2_gemm<cutlass::half_t, cutlass::gemm::GemmShape<16, 8, 8>,
                      cutlass::gemm::GemmShape<16, 64, 32>,
                      cutlass::gemm::GemmShape<64, 128, 64>,
                      cutlass::gemm::GemmShape<1024, 1024, 512>>();

    run_cutlass2_gemm<cutlass::half_t, cutlass::gemm::GemmShape<16, 8, 8>,
                      cutlass::gemm::GemmShape<32, 64, 32>,
                      cutlass::gemm::GemmShape<128, 128, 32>,
                      cutlass::gemm::GemmShape<2048, 4096, 512>>();

    // FIXME(ying): A case that cannot pass the unittest.
    // run_cutlass2_gemm<cutlass::half_t, cutlass::gemm::GemmShape<16, 8, 8>,
    //                   cutlass::gemm::GemmShape<64, 32, 32>,
    //                   cutlass::gemm::GemmShape<64, 96, 64>,
    //                   cutlass::gemm::GemmShape<128, 96, 64>>();
}

TEST(TestCuteGEMM, test2) {
    // The minimal valid shape
    run_cute_gemm<cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
                  TileShape<16, 32, 32>,               /*CTA Shape*/
                  TileShape<16, 32, 32>                /*Whole Shape*/
                  >();

    run_cute_gemm<cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
                  TileShape<16, 32, 32>,               /*CTA Shape*/
                  TileShape<16, 32, 64>                /*Whole Shape*/
                  >();

    run_cute_gemm<cutlass::half_t, TileShape<2, 2, 1>, /*Warp Arrangement*/
                  TileShape<64, 128, 64>,              /*CTA Shape*/
                  TileShape<128, 256, 512>             /*Whole Shape*/
                  >();
}

TEST(TestCuteGEMM, test_batched_gemm) {
    run_batched_cute_gemm<cutlass::half_t, TileShape<2, 2, 1>, /*Warp
                                                                 Arrangement*/
                          TileShape<64, 128, 64>,              /*CTA Shape*/
                          TileShape<128, 256, 512, 1>          /*Whole Shape*/
                          >();

    run_batched_cute_gemm<cutlass::half_t, TileShape<2, 2, 1>, /*Warp
                                           Arrangement*/
                          TileShape<64, 128, 64>,              /*CTA Shape*/
                          TileShape<128, 256, 512, 2>          /*Whole Shape*/
                          >();
    run_batched_cute_gemm<cutlass::half_t, TileShape<2, 2, 1>, /*Warp
                                         Arrangement*/
                          TileShape<64, 128, 64>,              /*CTA Shape*/
                          TileShape<128, 256, 512, 4>          /*Whole Shape*/
                          >();
}

}  // namespace core
}  // namespace kaleido
