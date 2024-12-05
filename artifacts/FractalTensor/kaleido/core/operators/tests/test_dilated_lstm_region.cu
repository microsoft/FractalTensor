#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/gemm.h"
#include "kaleido/core/device/kernels/lstm.h"
#include "kaleido/core/device/kernels/lstm/dilated_lstm/region1.h"
#include "kaleido/core/device/kernels/lstm/dilated_lstm/region2.h"
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
void run_cute_dilated_lstm_region1(const int depth, const int seq_length,
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
    kaleido::core::Tensor init({batch_size, hidden_size}, allocator);

    fill(xss, 0, 1e-3);
    fill(ws, 0, 1e-3);
    fill(us, 0, 1e-3);

    fill(init, 0.);
    fill(hsss, 0.);
    fill(csss, 0.);

    cuda_kernel::DilatedLstmRegion1<Element, InstructionShape, ValueMnk,
                                    WarpArrangement, CtaTileShape, WholeShape>(
        csss.mutable_data<Element>(), hsss.mutable_data<Element>(),
        xss.data<Element>(), ws.data<Element>(), us.data<Element>(),
        init.data<Element>(), seq_length);

    std::cout << "Pass unittest for [" << 4 * hidden_size << ", "
              << seq_length * batch_size << ", " << hidden_size << "]"
              << std::endl;
}

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_dilated_lstm_region2(const int depth, const int seq_length,
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

    cuda_kernel::DilatedLstmRegion2<Element, InstructionShape, ValueMnk,
                                    WarpArrangement, CtaTileShape>(
        csss.mutable_data<Element>(), hsss.mutable_data<Element>(),
        xss.data<Element>(), ws.data<Element>(), us.data<Element>(), depth,
        seq_length, batch_size, hidden_size);

    std::cout << "Pass unittest for [" << 4 * hidden_size << ", "
              << seq_length * batch_size << ", " << hidden_size << "]"
              << std::endl;
}

TEST(DilatedLstmRegion, test_dilated_lstm_region1) {
    // Define the problem size.
    const int depth = 4;
    const int seq_length = 4;
    const int batch_size = 8;
    const int hidden_size = 32;

    run_cute_dilated_lstm_region1<
        cutlass::half_t, TileShape<1, 1, 1>, TileShape<16, 32, 32>,
        TileShape<4 * hidden_size, seq_length * batch_size, hidden_size>>(
        depth, seq_length, batch_size, hidden_size);
}

TEST(DilatedLstmRegion, test_dilated_lstm_region2) {
    // Define the problem size.
    const int depth = 4;
    const int seq_length = 4;
    const int batch_size = 32;
    const int hidden_size = 32;

    run_cute_dilated_lstm_region2<cutlass::half_t, TileShape<1, 1, 1>,
                                  TileShape<16, 32, 32>>(
        depth, seq_length, batch_size, hidden_size);
}

}  // namespace core
}  // namespace kaleido
