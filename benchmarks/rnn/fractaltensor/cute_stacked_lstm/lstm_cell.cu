#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_timer.h"
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

#include <assert.h>
#include <cutlass/gemm/device/gemm.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace kaleido::core;

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void bench_cute_lstm_cell(const int depth, const int seq_length,
                          const int batch_size, const int hidden_size,
                          std::ofstream& fout) {
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

    int M = kM / 4;
    int N = kN;
    int K = kK;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor W({kM, kK}, allocator);
    kaleido::core::Tensor x_t({kK, kN}, allocator);
    kaleido::core::Tensor U({kM, kK}, allocator);
    kaleido::core::Tensor h_t_1({kK, kN}, allocator);
    kaleido::core::Tensor O({kM, kN}, allocator);

    kaleido::core::Tensor C_t_1({M, N}, allocator);
    kaleido::core::Tensor C_t({M, N}, allocator);
    kaleido::core::Tensor H_t({M, N}, allocator);

    fill(W, 0, 1e-3);
    fill(x_t, 0, 1e-3);
    fill(U, 0, 1e-3);
    fill(h_t_1, 0, 1e-3);

    fill(C_t_1, 0, 1e-3);
#ifdef DEBUG
    // fill(A, "seq");
    // fill(B, "seq");
#endif
    fill(O, 0.);

    fill(C_t, 0.);
    fill(H_t, 0.);

    using LSTMCell =
        cuda_kernel::CuteLSTMCell<Element, InstructionShape, ValueMnk,
                                  WarpArrangement, CtaTileShape, WholeShape>;
    LSTMCell lstm_cell;

    const int warm_up = 10;

    for (int i = 0; i < warm_up; ++i) {
        lstm_cell(W.data<Element>(), x_t.data<Element>(), U.data<Element>(),
                  C_t_1.data<Element>(), h_t_1.data<Element>(),
                  C_t.mutable_data<Element>(), H_t.mutable_data<Element>(),
                  O.mutable_data<Element>());
    }

    const int iter_counts = 100;
    float total_time = 0.f;

    for (int i = 0; i < iter_counts; ++i) {
        total_time +=
            lstm_cell(W.data<Element>(), x_t.data<Element>(), U.data<Element>(),
                      C_t_1.data<Element>(), h_t_1.data<Element>(),
                      C_t.mutable_data<Element>(), H_t.mutable_data<Element>(),
                      O.mutable_data<Element>());
    }

    fout << "[" << depth << ", " << seq_length << ", " << batch_size << ", "
         << hidden_size << "]"
         << "\t"
         << "[" << kTM << ", " << kTN << ", " << kTK << "]"
         << "\t" << total_time / iter_counts << std::endl;
}

int main(int argc, char* argv[]) {
    assert(argc == 2);
    const char* filename = argv[1];

    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    fout.open(filename, std::ios::out);

    // The minimal valid shape
    // Whole Shape: [4 * hidden_size, batch_size, hidden_size]
    std::cout << "Cute LstmCell Benchmark......" << std::endl;
    fout << "[depth, seq_length, batch_size, hidden_size]\t[kTM, kTN, "
            "kTK]\tAvgTime(ms)"
         << std::endl;

    constexpr std::array<int, 4> hidden_sizes = {128, 256, 512, 1024};
    constexpr std::array<int, 4> batch_sizes = {32, 64, 128, 256};

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[0], batch_sizes[0],
                                   hidden_sizes[0]> /*Whole Shape*/
                         >(1, 1, batch_sizes[0], hidden_sizes[0], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[0], batch_sizes[1],
                                   hidden_sizes[0]> /*Whole Shape*/
                         >(1, 1, batch_sizes[1], hidden_sizes[0], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[0], batch_sizes[2],
                                   hidden_sizes[0]> /*Whole Shape*/
                         >(1, 1, batch_sizes[2], hidden_sizes[0], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[0], batch_sizes[3],
                                   hidden_sizes[0]> /*Whole Shape*/
                         >(1, 1, batch_sizes[3], hidden_sizes[0], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[1], batch_sizes[0],
                                   hidden_sizes[1]> /*Whole Shape*/
                         >(1, 1, batch_sizes[0], hidden_sizes[1], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[1], batch_sizes[1],
                                   hidden_sizes[1]> /*Whole Shape*/
                         >(1, 1, batch_sizes[1], hidden_sizes[1], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[1], batch_sizes[2],
                                   hidden_sizes[1]> /*Whole Shape*/
                         >(1, 1, batch_sizes[2], hidden_sizes[1], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[1], batch_sizes[3],
                                   hidden_sizes[1]> /*Whole Shape*/
                         >(1, 1, batch_sizes[3], hidden_sizes[1], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[2], batch_sizes[0],
                                   hidden_sizes[2]> /*Whole Shape*/
                         >(1, 1, batch_sizes[0], hidden_sizes[2], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[2], batch_sizes[1],
                                   hidden_sizes[2]> /*Whole Shape*/
                         >(1, 1, batch_sizes[1], hidden_sizes[2], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[2], batch_sizes[2],
                                   hidden_sizes[2]> /*Whole Shape*/
                         >(1, 1, batch_sizes[2], hidden_sizes[2], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 2, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[2], batch_sizes[3],
                                   hidden_sizes[2]> /*Whole Shape*/
                         >(1, 1, batch_sizes[3], hidden_sizes[2], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[3], batch_sizes[0],
                                   hidden_sizes[3]> /*Whole Shape*/
                         >(1, 1, batch_sizes[0], hidden_sizes[3], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[3], batch_sizes[1],
                                   hidden_sizes[3]> /*Whole Shape*/
                         >(1, 1, batch_sizes[1], hidden_sizes[3], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[3], batch_sizes[2],
                                   hidden_sizes[3]> /*Whole Shape*/
                         >(1, 1, batch_sizes[2], hidden_sizes[3], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<1, 1, 1>,    /*Warp Arrangement*/
                         TileShape<16, 32, 32>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[3], batch_sizes[3],
                                   hidden_sizes[3]> /*Whole Shape*/
                         >(1, 1, batch_sizes[3], hidden_sizes[3], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[0], batch_sizes[0],
                                   hidden_sizes[0]> /*Whole Shape*/
                         >(1, 1, batch_sizes[0], hidden_sizes[0], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[0], batch_sizes[1],
                                   hidden_sizes[0]> /*Whole Shape*/
                         >(1, 1, batch_sizes[1], hidden_sizes[0], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[0], batch_sizes[2],
                                   hidden_sizes[0]> /*Whole Shape*/
                         >(1, 1, batch_sizes[2], hidden_sizes[0], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[0], batch_sizes[3],
                                   hidden_sizes[0]> /*Whole Shape*/
                         >(1, 1, batch_sizes[3], hidden_sizes[0], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[1], batch_sizes[0],
                                   hidden_sizes[1]> /*Whole Shape*/
                         >(1, 1, batch_sizes[0], hidden_sizes[1], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[1], batch_sizes[1],
                                   hidden_sizes[1]> /*Whole Shape*/
                         >(1, 1, batch_sizes[1], hidden_sizes[1], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[1], batch_sizes[2],
                                   hidden_sizes[1]> /*Whole Shape*/
                         >(1, 1, batch_sizes[2], hidden_sizes[1], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[1], batch_sizes[3],
                                   hidden_sizes[1]> /*Whole Shape*/
                         >(1, 1, batch_sizes[3], hidden_sizes[1], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[2], batch_sizes[0],
                                   hidden_sizes[2]> /*Whole Shape*/
                         >(1, 1, batch_sizes[0], hidden_sizes[2], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[2], batch_sizes[1],
                                   hidden_sizes[2]> /*Whole Shape*/
                         >(1, 1, batch_sizes[1], hidden_sizes[2], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[2], batch_sizes[2],
                                   hidden_sizes[2]> /*Whole Shape*/
                         >(1, 1, batch_sizes[2], hidden_sizes[2], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[2], batch_sizes[3],
                                   hidden_sizes[2]> /*Whole Shape*/
                         >(1, 1, batch_sizes[3], hidden_sizes[2], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[3], batch_sizes[0],
                                   hidden_sizes[3]> /*Whole Shape*/
                         >(1, 1, batch_sizes[0], hidden_sizes[3], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[3], batch_sizes[1],
                                   hidden_sizes[3]> /*Whole Shape*/
                         >(1, 1, batch_sizes[1], hidden_sizes[3], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[3], batch_sizes[2],
                                   hidden_sizes[3]> /*Whole Shape*/
                         >(1, 1, batch_sizes[2], hidden_sizes[3], fout);

    bench_cute_lstm_cell<cutlass::half_t,
                         TileShape<2, 2, 1>,     /*Warp Arrangement*/
                         TileShape<64, 128, 64>, /*CTA Shape*/
                         TileShape<4 * hidden_sizes[3], batch_sizes[3],
                                   hidden_sizes[3]> /*Whole Shape*/
                         >(1, 1, batch_sizes[3], hidden_sizes[3], fout);
}
