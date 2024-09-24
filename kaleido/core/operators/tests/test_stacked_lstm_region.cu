#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/gemm.h"
#include "kaleido/core/device/kernels/lstm.h"
#include "kaleido/core/device/kernels/lstm/stacked_lstm/region1.h"
#include "kaleido/core/device/kernels/lstm/stacked_lstm/region2.h"
#include "kaleido/core/device/kernels/lstm/stacked_lstm/region3.h"
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

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_stacked_lstm_region1(const int depth, const int seq_length,
                                   const int batch_size,
                                   const int hidden_size) {
    // Initialize the CUDA context.
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
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor ws({depth, hidden_size, 4 * hidden_size}, allocator);
    kaleido::core::Tensor us({depth, hidden_size, 4 * hidden_size}, allocator);
    kaleido::core::Tensor xss({seq_length, batch_size, hidden_size}, allocator);
    kaleido::core::Tensor hsss({depth, seq_length, batch_size, hidden_size},
                               allocator);
    kaleido::core::Tensor csss({depth, seq_length, batch_size, hidden_size},
                               allocator);

    // Fill the input tensors with random values.
    fill(xss, 0, 1e-3);
    fill(ws, 0, 1e-3);
    fill(us, 0, 1e-3);

    // Fill the output tensors with zero.
    fill(hsss, 0.);
    fill(csss, 0.);

    // Run the our LSTM implementation.
    float time = cuda_kernel::StackedLstmRegion1<Element, InstructionShape,
                                                 ValueMnk, WarpArrangement,
                                                 CtaTileShape, WholeShape>(
        hsss.mutable_data<Element>(), csss.mutable_data<Element>(),
        xss.data<Element>(), ws.data<Element>(), us.data<Element>(), depth,
        seq_length, batch_size, hidden_size);

    std::cout << "Pass unittest for [" << 4 * hidden_size << ", " << batch_size
              << ", " << hidden_size << "]" << std::endl;

    std::cout << "Time: " << time << " ms" << std::endl;
}

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_stacked_lstm_region2(const int depth, const int seq_length,
                                   const int batch_size,
                                   const int hidden_size) {
    // Initialize the CUDA context.
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
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor ws({depth, hidden_size, 4 * hidden_size}, allocator);
    kaleido::core::Tensor us({depth, hidden_size, 4 * hidden_size}, allocator);
    kaleido::core::Tensor xss({seq_length, batch_size, hidden_size}, allocator);
    kaleido::core::Tensor hsss({depth, seq_length, batch_size, hidden_size},
                               allocator);
    kaleido::core::Tensor csss({depth, seq_length, batch_size, hidden_size},
                               allocator);

    // Fill the input tensors with random values.
    fill(xss, 0, 1e-3);
    fill(ws, 0, 1e-3);
    fill(us, 0, 1e-3);

    // Fill the output tensors with zero.
    fill(hsss, 0.);
    fill(csss, 0.);

    float time = cuda_kernel::StackedLstmRegion2<Element, InstructionShape,
                                                 ValueMnk, WarpArrangement,
                                                 CtaTileShape, WholeShape>(
        hsss.mutable_data<Element>(), csss.mutable_data<Element>(),
        xss.data<Element>(), ws.data<Element>(), us.data<Element>(), depth,
        seq_length, batch_size, hidden_size);

    std::cout << "Pass unittest for [" << 4 * hidden_size << ", "
              << seq_length * batch_size << ", " << hidden_size << "]"
              << std::endl;

    std::cout << "Time: " << time << " ms" << std::endl;
}

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_stacked_lstm_region3(const int depth, const int seq_length,
                                   const int batch_size,
                                   const int hidden_size) {
    // Initialize the CUDA context.
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
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor ws({depth, hidden_size, 4 * hidden_size}, allocator);
    kaleido::core::Tensor us({depth, hidden_size, 4 * hidden_size}, allocator);
    kaleido::core::Tensor xss({seq_length, batch_size, hidden_size}, allocator);
    kaleido::core::Tensor hsss({depth, seq_length, batch_size, hidden_size},
                               allocator);
    kaleido::core::Tensor csss({depth, seq_length, batch_size, hidden_size},
                               allocator);

    // Fill the input tensors with random values.
    fill(xss, 0, 1e-3);
    fill(ws, 0, 1e-3);
    fill(us, 0, 1e-3);

    // Fill the output tensors with zero.
    fill(hsss, 0.);
    fill(csss, 0.);

    // Run the our LSTM implementation.

    float time =
        cuda_kernel::StackedLstmRegion3<Element, InstructionShape, ValueMnk,
                                        WarpArrangement, CtaTileShape>(
            hsss.mutable_data<Element>(), csss.mutable_data<Element>(),
            xss.data<Element>(), ws.data<Element>(), us.data<Element>(), depth,
            seq_length, batch_size, hidden_size);

    std::cout << "Pass unittest for [" << 4 * hidden_size << ", "
              << seq_length * batch_size << ", " << hidden_size << "]"
              << std::endl;

    std::cout << "Time: " << time << " ms" << std::endl;
}

TEST(StackedLstmRegions, test_stacked_lstm_region1) {
    run_cute_stacked_lstm_region1<cutlass::half_t,
                                  TileShape<1, 1, 1>,    /*Warp Arrangement*/
                                  TileShape<16, 32, 32>, /*CTA Shape*/
                                  TileShape<4 * 32, 4 * 8,
                                            32> /*Whole
                   Shape*/>(4, 4, 8, 32);

    run_cute_stacked_lstm_region1<
        cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
        TileShape<16, 32, 32>,               /*CTA Shape*/
        TileShape<4 * 64, 16 * 32, 64> /*Whole Shape*/>(2, 16, 32, 64);

    run_cute_stacked_lstm_region1<cutlass::half_t,
                                  TileShape<1, 1, 1>,    /*Warp Arrangement*/
                                  TileShape<16, 32, 32>, /*CTA Shape*/
                                  TileShape<4 * 64, 32, 64> /*Whole Shape*/>(
        2, 32, 32, 64);

    run_cute_stacked_lstm_region1<cutlass::half_t,
                                  TileShape<1, 1, 1>,    /*Warp Arrangement*/
                                  TileShape<16, 32, 32>, /*CTA Shape*/
                                  TileShape<4 * 64, 32, 64> /*Whole Shape*/>(
        2, 50, 32, 64);

    run_cute_stacked_lstm_region1<cutlass::half_t,
                                  TileShape<1, 1, 1>,    /*Warp Arrangement*/
                                  TileShape<16, 32, 32>, /*CTA Shape*/
                                  TileShape<4 * 128, 64, 128> /*Whole Shape*/>(
        16, 64, 64, 128);

    // Too large shape.
    // run_cute_stacked_lstm_region1<cutlass::half_t,
    //                               TileShape<1, 1, 1>,    /*Warp Arrangement*/
    //                               TileShape<16, 32, 32>, /*CTA Shape*/
    //                               TileShape<4 * 256, 64,
    //                                         256> /*Whole
    //                  Shape*/>(16, 64, 64, 256);
}

TEST(StackedLstmRegions, test_stacked_lstm_region2) {
    run_cute_stacked_lstm_region2<
        cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
        TileShape<16, 32, 32>,               /*CTA Shape*/
        TileShape<4 * 32, 4 * 32, 32> /*Whole Shape*/>(4, 4, 32, 32);

    run_cute_stacked_lstm_region2<
        cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
        TileShape<16, 32, 32>,               /*CTA Shape*/
        TileShape<4 * 64, 16 * 32, 64> /*Whole Shape*/>(2, 16, 32, 64);

    run_cute_stacked_lstm_region2<
        cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
        TileShape<16, 32, 32>,               /*CTA Shape*/
        TileShape<4 * 64, 32 * 32, 64> /*Whole Shape*/>(2, 32, 32, 64);

    run_cute_stacked_lstm_region2<
        cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
        TileShape<16, 32, 32>,               /*CTA Shape*/
        TileShape<4 * 64, 50 * 32, 64> /*Whole Shape*/>(2, 50, 32, 64);
}

TEST(StackedLstmRegions, test_stacked_lstm_region3) {
    run_cute_stacked_lstm_region3<cutlass::half_t,
                                  TileShape<1, 1, 1>, /*Warp Arrangement*/
                                  TileShape<16, 32, 32> /*CTA Shape*/>(4, 4, 32,
                                                                       32);
    run_cute_stacked_lstm_region3<cutlass::half_t,
                                  TileShape<1, 1, 1>, /*Warp Arrangement*/
                                  TileShape<16, 32, 32> /*CTA Shape*/>(2, 16,
                                                                       32, 64);
    run_cute_stacked_lstm_region3<cutlass::half_t,
                                  TileShape<1, 1, 1>, /*Warp Arrangement*/
                                  TileShape<16, 32, 32> /*CTA Shape*/>(2, 32,
                                                                       32, 64);
    run_cute_stacked_lstm_region3<cutlass::half_t,
                                  TileShape<1, 1, 1>, /*Warp Arrangement*/
                                  TileShape<16, 32, 32> /*CTA Shape*/>(2, 64,
                                                                       32, 64);

    // TODO: irregular seq_len will cause illegal memory access.
    // run_cute_stacked_lstm_region3<cutlass::half_t,
    //                               TileShape<1, 1, 1>, /*Warp Arrangement*/
    //                               TileShape<16, 32, 32> /*CTA Shape*/>(2, 50,
    //                                                                    32,
    //                                                                    64);
}

}  // namespace core
}  // namespace kaleido
