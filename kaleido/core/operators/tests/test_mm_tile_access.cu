#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/gmem_tile_transmitter.h"
#include "kaleido/core/device/kernels/smem_tile_transmitter.h"
#include "kaleido/core/operators/expect_eq_op.h"
#include "kaleido/core/operators/tests/test_utils.h"

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <gtest/gtest.h>

namespace kaleido {
namespace core {
namespace ops {

using namespace test_utils;
using namespace cuda_kernel;

namespace {
template <const int kThreads, typename Element, const int kM, const int kN,
          const int kK, const int kTM, const int kTN, const int kTK,
          typename LoadA, typename StoreA, typename LoadB, typename StoreB,
          typename LoadC, typename StoreC>
__global__ void TestCopy(Element* dA, Element* dA_copy, Element* dB,
                         Element* dB_copy, Element* dC, Element* dC_copy,
                         LoadA loader_a, StoreA storer_a, LoadB loader_b,
                         StoreB storer_b, LoadC loader_c, StoreC storer_c) {
    __shared__ cutlass::AlignedBuffer<Element, kTM * kTK> smem_buf_a;
    __shared__ cutlass::AlignedBuffer<Element, kTK * kTN> smem_buf_b;

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // advance the start ptr of A, B, C matrix for the current CTA
    Element* A = dA + cRow * kTM * kK;
    Element* A_copy = dA_copy + cRow * kTM * kK;

    Element* B = dB + cCol * kTN;
    Element* B_copy = dB_copy + cCol * kTN;

    Element* C = dC + cRow * kTM * kN + cCol * kTN;
    Element* C_copy = dC_copy + cRow * kTM * kN + cCol * kTN;

    // simulate matrix C is stored on the shared memory also.
    __shared__ cutlass::AlignedBuffer<Element, kTM * kTN> smem_buf_c;
    loader_c.template transfer(C, smem_buf_c.data(), kN, threadIdx.x);
    // Test store C block on shared memory into global memory.
    storer_c.template transfer(smem_buf_c.data(), C_copy, kN, threadIdx.x);

    // Advance over the K dimension
    for (int bkIdx = 0; bkIdx < kK; bkIdx += kTK) {
        loader_a.template transfer(A, smem_buf_a.data(), kK, threadIdx.x);
        storer_a.template transfer(smem_buf_a.data(), A_copy, kK, threadIdx.x);

        loader_b.template transfer(B, smem_buf_b.data(), kN, threadIdx.x);
        storer_b.template transfer(smem_buf_b.data(), B_copy, kN, threadIdx.x);

        A += kTK;  // advance pointer
        A_copy += kTK;

        B += kTK * kN;
        B_copy += kTK * kN;
    }
}

template <const int kThreads, typename Element, const int kM, const int kN,
          const int kK, const int kTM, const int kTN, const int kTK>
int TestGemmAccess(const Element* dA, Element* dA_copy, const Element* dB,
                   Element* dB_copy, const Element* dC, Element* dC_copy) {
    using LoadA =
        GmemTileTransmitter<kTM, kTK, Element, kThreads, TileLayout::RowMajor,
                            TileLayout::SwizzledRowMajor>;
    using StoreA =
        SmemTileTransmitter<kTM, kTK, Element, kThreads,
                            TileLayout::SwizzledRowMajor, TileLayout::RowMajor>;
    LoadA loader_a(kTM, kTK);
    StoreA storer_a(kTM, kTK);

    using LoadB =
        GmemTileTransmitter<kTK, kTN, Element, kThreads, TileLayout::RowMajor,
                            TileLayout::SwizzledColumnMajor>;
    // the stored data is TRANSPOSE of the input data
    using StoreB = SmemTileTransmitter<kTN, kTK, Element, kThreads,
                                       TileLayout::SwizzledColumnMajor,
                                       TileLayout::RowMajor>;
    LoadB loader_b(kTK, kTN);
    StoreB storer_b(kTN, kTK);

    using LoadC =
        GmemTileTransmitter<kTM, kTN, Element, kThreads, TileLayout::RowMajor,
                            TileLayout::RowMajor>;
    using StoreC =
        SmemTileTransmitter<kTM, kTN, Element, kThreads, TileLayout::RowMajor,
                            TileLayout::RowMajor>;
    LoadC loader_c(kTM, kTN);
    StoreC storer_c(kTM, kTN);

    const int block_m = DIVUP(kM, kTM);
    const int block_n = DIVUP(kN, kTN);
    std::cout << "blocks = [" << block_m << "," << block_n << "]" << std::endl;

    dim3 gridDim(block_m, block_n);
    dim3 blockDim(kThreads, 1, 1);
    TestCopy<kThreads, Element, kM, kN, kK, kTM, kTN, kTK, decltype(loader_a),
             decltype(storer_a), decltype(loader_b), decltype(storer_b),
             decltype(loader_c), decltype(storer_c)><<<gridDim, blockDim>>>(
        const_cast<Element*>(dA), dA_copy, const_cast<Element*>(dB), dB_copy,
        const_cast<Element*>(dC), dC_copy, loader_a, storer_a, loader_b,
        storer_b, loader_c, storer_c);
    return 0;
}

}  // namespace

template <typename Element, typename WholeShape, typename TileShape,
          const int kThreads>
void run_test() {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    std::shared_ptr<Tensor> dA = SequentialGpuTensor<Element>(
        {WholeShape::kM, WholeShape::kK}, allocator);
    std::shared_ptr<Tensor> dA_copy = ConstantGpuTensor<Element>(
        {WholeShape::kM, WholeShape::kK}, allocator, static_cast<Element>(0));

    std::shared_ptr<Tensor> dB = SequentialGpuTensor<Element>(
        {WholeShape::kK, WholeShape::kN}, allocator);
    std::shared_ptr<Tensor> dB_copy = ConstantGpuTensor<Element>(
        {WholeShape::kK, WholeShape::kN}, allocator, 0.);

    std::shared_ptr<Tensor> dC = SequentialGpuTensor<Element>(
        {WholeShape::kM, WholeShape::kN}, allocator);

    std::shared_ptr<Tensor> dC_copy = ConstantGpuTensor<Element>(
        {WholeShape::kM, WholeShape::kN}, allocator, 0.);

    TestGemmAccess<kThreads, Element, WholeShape::kM, WholeShape::kN,
                   WholeShape::kK, TileShape::kM, TileShape::kN, TileShape::kK>(
        dA->data<Element>(), dA_copy->mutable_data<Element>(),
        dB->data<Element>(), dB_copy->mutable_data<Element>(),
        dC->data<Element>(), dC_copy->mutable_data<Element>());

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(*dA, *dA_copy, 1e-5);
    check(*dB, *dB_copy, 1e-5);
    check(*dC, *dC_copy, 1e-5);
}

TEST(TestGemmTileAccess, test) {
    run_test<float, cutlass::gemm::GemmShape<112, 96, 384>,
             cutlass::gemm::GemmShape<16, 32, 128>, 128>();

    run_test<cutlass::half_t, cutlass::gemm::GemmShape<48, 160, 128>,
             cutlass::gemm::GemmShape<16, 32, 64>, 64>();
}

}  // namespace ops
}  // namespace core
}  // namespace kaleido
