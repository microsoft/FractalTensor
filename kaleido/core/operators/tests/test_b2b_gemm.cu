#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/gemm.h"
#include "kaleido/core/operators/expect_eq_op.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/print_op.h"
#include "kaleido/core/operators/tests/b2b_gemm_test_utils.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tile_shape.h"

#include <gtest/gtest.h>

using namespace kaleido::core;

template <typename Element, typename WholeShape, typename CtaTileShape,
          typename WarpShape>
void run_test() {
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

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;

    const int kM = dim_size<0, WholeShape>;
    const int kN = dim_size<1, WholeShape>;
    const int kK = dim_size<2, WholeShape>;
    const int kP = dim_size<3, WholeShape>;

    const int kTM = dim_size<0, CtaTileShape>;
    const int kTN = dim_size<1, CtaTileShape>;
    const int kTK = dim_size<2, CtaTileShape>;
    const int kTP = dim_size<3, CtaTileShape>;

    const int num_warps = dim_size<0, WarpShape>;

    kaleido::core::Tensor A({kM, kK}, allocator);
    kaleido::core::Tensor B({kK, kN}, allocator);
    kaleido::core::Tensor C({kN, kP}, allocator);
    kaleido::core::Tensor D({kM, kP}, allocator);

    kaleido::core::Tensor ref_P({kM, kN}, allocator);
    kaleido::core::Tensor ref_D({kM, kP}, allocator);

    fill(A, 0., 1e-3);
    fill(B, 0., 1e-3);
    fill(C, 0., 1e-3);

    fill(D, 0.);
    fill(ref_P, 0.);
    fill(ref_D, 0.);

    using Gemm =
        cuda_kernel::B2BGemm<Element, WholeShape, CtaTileShape, WarpShape>;
    Gemm gemm;

    gemm(A.data<Element>(), B.data<Element>(), C.data<Element>(),
         D.mutable_data<Element>());

    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));

    cublas_two_hgemms<Element>(handle, A, B, C,  // inputs
                               ref_P,            // ref_P = A @ B
                               ref_D /*ref_D = ref_P @ C*/);

    CublasCheck(cublasDestroy(handle));

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    LOG(INFO) << "blocks: [" << CeilDiv<kM, kTM> << ", "
              << CeilDiv<kP, kTP> << "], threads: " << num_warps * 32
              << std::endl;
    LOG(INFO) << "Problem shape:\t[" << kM << ", " << kK << "] @ [" << kK << ","
              << kN << "] @ [" << kN << ", " << kP << "]" << std::endl;
    LOG(INFO) << "CTA tile shape:\t[" << kTM << ", " << kTK << "] @ [" << kTK
              << ", " << kTN << "] @ [" << kTN << ", " << kTP << "]"
              << std::endl;

    try {  // check correctness
        check(D, ref_D, 3e-3);

        LOG(INFO) << "D ref: " << std::endl << printer(ref_D, 4, 20, 27);
        LOG(INFO) << "D: " << std::endl << printer(D, 4, 20 /*count*/, 27);

        LOG(INFO) << "Passed Unittest." << std::endl << std::endl;
    } catch (const std::invalid_argument& e) {
        LOG(INFO) << "D ref: " << std::endl << printer(ref_D, 4, 20, 27);
        LOG(INFO) << "D: " << std::endl << printer(D, 4, 20 /*count*/, 27);
        LOG(INFO) << "Failed Unittest." << std::endl << std::endl;
    }
}

TEST(TestB2BGemm, test_b2b_gemm) {
    FLAGS_logtostderr = 1;

    // The minimal shape for back-2-back GEMM
    // TiledMMA: [16, 16, 16]
    run_test<cutlass::half_t,
             TileShape<32 /*kM*/, 32 /*kN*/, 32 /*kK*/, 32 /*kP*/>,
             TileShape<32 /*kTM*/, 32 /*kTN*/, 32 /*kTK*/, 32 /*kTP*/>,
             TileShape<2, 1> /*warp arrangement of the CTA*/
             >();

    // TiledMMA: [16, 16, 16]
    run_test<cutlass::half_t,
             TileShape<16 /*kM*/, 64 /*kN*/, 64 /*kK*/, 64 /*kP*/>,
             TileShape<16 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
             TileShape<1, 1> /*warp arrangement of the CTA*/
             >();

    // test shared memory swizzled plan for shared memory tile size other
    // than 32. TiledMMA: [64, 16, 16]
    run_test<cutlass::half_t,
             TileShape<32 /*kM*/, 64 /*kN*/, 64 /*kK*/, 64 /*kP*/>,
             TileShape<32 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
             TileShape<2, 1> /*warp arrangement of the CTA*/
             >();

    run_test<cutlass::half_t,
             TileShape<64 /*kM*/, 64 /*kN*/, 64 /*kK*/, 64 /*kP*/>,
             TileShape<64 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
             TileShape<4, 1> /*warp arrangement of the CTA*/
             >();

    run_test<
        cutlass::half_t,
        TileShape<64 * 3 /*kM*/, 64 * 7 /*kN*/, 64 * 11 /*kK*/, 64 * 13 /*kP*/>,
        TileShape<64 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
        TileShape<4, 1> /*warp arrangement of the CTA*/
        >();

    run_test<
        cutlass::half_t,
        TileShape<64 * 3 /*kM*/, 64 * 7 /*kN*/, 64 * 11 /*kK*/, 64 * 13 /*kP*/>,
        TileShape<16 /*kTM*/, 128 /*kTN*/, 128 /*kTK*/, 128 /*kTP*/>,
        TileShape<1, 1> /*warp arrangement of the CTA*/
        >();
    // Add more tile shape examples.
}
