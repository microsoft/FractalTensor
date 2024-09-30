#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/gemm.h"
#include "kaleido/core/device/kernels/lstm.h"
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
          typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_dynamic_fused_batched_lstm_cell(const int m, const int n, const int k,
                                         const int b, int hidden_size,
                                         int batch_size, int seq_length,
                                         int depth) {
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
    int kM = m;
    int kN = n;
    int kK = k;
    int kB = b;

    int M = kM / 4;
    int N = kN;
    int K = kK;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor ws({depth, hidden_size, 4 * hidden_size}, allocator);
    kaleido::core::Tensor us({depth, hidden_size, 4 * hidden_size}, allocator);
    kaleido::core::Tensor xss({depth, seq_length, batch_size, hidden_size},
                              allocator);
    kaleido::core::Tensor hsss({depth, seq_length, batch_size, hidden_size},
                               allocator);
    kaleido::core::Tensor csss({depth, seq_length, batch_size, hidden_size},
                               allocator);
    kaleido::core::Tensor c_out({depth, seq_length, batch_size, hidden_size},
                                allocator);
    kaleido::core::Tensor h_out({depth, seq_length, batch_size, hidden_size},
                                allocator);

    // Fill the input tensors with random values.
    fill(xss, 0, 1e-3);
    fill(ws, 0, 1e-3);
    fill(us, 0, 1e-3);

    // Fill the output tensors with zero.
    fill(hsss, 0.);
    fill(csss, 0.);
    fill(c_out, 0.);
    fill(h_out, 0.);

    const int stride_a = m * k;
    const int stride_b = k * n;
    const int stride_c = m * n;

    cuda_kernel::CuteFusedBMMLSTMCell<Element, InstructionShape, ValueMnk,
                                      WarpArrangement, CtaTileShape>
        cute_fused_bmm_lstm_cell;

    cute_fused_bmm_lstm_cell(
        ws.data<Element>(), xss.data<Element>(), us.data<Element>(),
        csss.data<Element>(), hsss.data<Element>(),
        c_out.mutable_data<Element>(), h_out.mutable_data<Element>(), depth,
        seq_length, hidden_size, b, stride_a, stride_b, stride_c, m, n, k, b);

    std::cout << "Passed unittest for [" << kM << ", " << kN << ", " << kK
              << "]" << std::endl;
}

TEST(DynamicBatchedLstmCell, test_dynamic_batched_lstm_cell) {
    // Define the problem size.
    const int depth = 4;
    const int seq_length = 4;
    const int batch_size = 32;
    const int hidden_size = 32;

    const int m = 4 * hidden_size;
    const int n = batch_size * seq_length;
    const int k = hidden_size;
    const int b = 1;

    run_dynamic_fused_batched_lstm_cell<cutlass::half_t,
                                        TileShape<1, 1, 1>, /*Warp Arrangement*/
                                        TileShape<16, 32, 32>>(
        m, m, k, b, depth, seq_length, batch_size, hidden_size);
}
}  // namespace core
}  // namespace kaleido
