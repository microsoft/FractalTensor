#include "cuDNN/utils.h"
#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/fill.h"
#include "kaleido/core/device/kernels/lstm.h"
#include "kaleido/core/device/kernels/lstm/dilated_lstm/region1.h"
#include "kaleido/core/device/kernels/lstm/dilated_lstm/region2.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/print_op.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"

#include <assert.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

using namespace kaleido::core;

float bench_cudnn_lstm_half(int mini_batch, int hidden_size, int seq_length,
                            int num_layers, int input_size) {
    RNNSampleOptions options;

    options.dataType = 0;  //   1 for float, 0 for half
    options.seqLength = seq_length;
    options.numLayers = num_layers;
    options.inputSize = input_size;
    options.hiddenSize = hidden_size;
    options.projSize = hidden_size;
    options.miniBatch = mini_batch;
    options.inputMode = 1;  // CUDNN_LINEAR_INPUT
    options.dirMode = 0;    // CUDNN_UNIDIRECTIONAL
    options.cellMode = 2;   // CUDNN_LSTM
    options.biasMode = 3;   // CUDNN_RNN_DOUBLE_BIAS
    options.algorithm = 0;  // CUDNN_RNN_ALGO_STANDARD

    options.mathPrecision = 0;  //  1 for float, 0 for half

    options.mathType = 1;  // CUDNN_TENSOR_OP_MATH
    options.dropout = 0.;
    options.printWeights = 0;

    return runRNNSample<__half>(options);
}

template <typename Element, typename WarpArragement, typename CtaTileShape,
          typename WholeShape, typename CellShape2,
          typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
float run_dilated_region1(const int depth, const int seq_length,
                          const int batch_size, const int hidden_size,
                          std::ofstream& fout) {
    size_t numel = depth * 4 * hidden_size * hidden_size;
    Element* dWs;
    CudaCheck(cudaMalloc(&dWs, numel * sizeof(Element)));
    cuda_kernel::FillRandomValue(dWs, numel);

    numel = seq_length * batch_size * hidden_size;
    Element* dXs;
    CudaCheck(cudaMalloc(&dXs, numel * sizeof(Element)));
    cuda_kernel::FillRandomValue(dXs, numel);

    numel = depth * 4 * hidden_size * hidden_size;
    Element* dUs;
    CudaCheck(cudaMalloc(&dUs, numel * sizeof(Element)));
    cuda_kernel::FillRandomValue(dUs, numel);

    numel = depth * seq_length * batch_size * hidden_size;
    Element* dCsss;
    CudaCheck(cudaMalloc(&dCsss, numel * sizeof(Element)));
    CudaCheck(cudaMemset(dCsss, 0, numel * sizeof(Element)));

    Element* dHsss;
    CudaCheck(cudaMalloc(&dHsss, numel * sizeof(Element)));
    CudaCheck(cudaMemset(dHsss, 0, numel * sizeof(Element)));

    numel = seq_length * batch_size * hidden_size;
    Element* init;
    CudaCheck(cudaMalloc(&init, numel * sizeof(Element)));
    cuda_kernel::FillRandomValue(init, numel);

    const int repeat = 10;

    float elapsed_time = 0.0f;

    for (auto i = 0; i < repeat; i++) {
        elapsed_time +=
            cuda_kernel::DilatedLstmRegion1<Element, InstructionShape, ValueMnk,
                                            WarpArragement, CtaTileShape,
                                            WholeShape>(dHsss, dCsss, dXs, dWs,
                                                        dUs, init, seq_length);
    }

    CudaCheck(cudaFree(dWs));
    CudaCheck(cudaFree(dXs));
    CudaCheck(cudaFree(dUs));
    CudaCheck(cudaFree(dCsss));
    CudaCheck(cudaFree(dHsss));
    CudaCheck(cudaFree(init));

    float avg_time = elapsed_time / repeat;

    return avg_time;
}

template <typename Element, typename WarpArragement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
float run_dilated_lstm_cell(const int depth, const int seq_length,
                            const int batch_size, const int hidden_size,
                            int iter_count, std::ofstream& fout) {
    size_t numel = depth * 4 * hidden_size * hidden_size;
    Element* dWs;
    CudaCheck(cudaMalloc(&dWs, numel * sizeof(Element)));
    cuda_kernel::FillRandomValue(dWs, numel);

    numel = seq_length * batch_size * hidden_size;
    Element* dXs;
    CudaCheck(cudaMalloc(&dXs, numel * sizeof(Element)));
    cuda_kernel::FillRandomValue(dXs, numel);

    numel = depth * 4 * hidden_size * hidden_size;
    Element* dUs;
    CudaCheck(cudaMalloc(&dUs, numel * sizeof(Element)));
    cuda_kernel::FillRandomValue(dUs, numel);

    numel = depth * seq_length * batch_size * hidden_size;
    Element* dCsss;
    CudaCheck(cudaMalloc(&dCsss, numel * sizeof(Element)));
    CudaCheck(cudaMemset(dCsss, 0, numel * sizeof(Element)));

    Element* dHsss;
    CudaCheck(cudaMalloc(&dHsss, numel * sizeof(Element)));
    CudaCheck(cudaMemset(dHsss, 0, numel * sizeof(Element)));

    numel = seq_length * batch_size * hidden_size;
    Element* init;
    CudaCheck(cudaMalloc(&init, numel * sizeof(Element)));
    cuda_kernel::FillRandomValue(init, numel);

    numel = 4 * 256 * batch_size * hidden_size;
    Element* c_out;
    CudaCheck(cudaMalloc(&c_out, numel * sizeof(Element)));

    Element* h_out;
    CudaCheck(cudaMalloc(&h_out, numel * sizeof(Element)));

    Element* t;
    CudaCheck(cudaMalloc(&t, numel * sizeof(Element)));

    cuda_kernel::CuteLSTMCell<Element, InstructionShape, ValueMnk,
                              WarpArragement, CtaTileShape, WholeShape>
        lstm_cell;
    float elapsed_time = 0.0f;

    for (int i = 0; i < iter_count; i++) {
        elapsed_time += lstm_cell(dWs, dXs, dUs, dCsss, dHsss, c_out, h_out, t);
    }

    CudaCheck(cudaFree(dWs));
    CudaCheck(cudaFree(dXs));
    CudaCheck(cudaFree(dUs));
    CudaCheck(cudaFree(dCsss));
    CudaCheck(cudaFree(dHsss));
    CudaCheck(cudaFree(init));
    CudaCheck(cudaFree(c_out));
    CudaCheck(cudaFree(h_out));
    CudaCheck(cudaFree(t));

    return elapsed_time;
}

template <int depth, int seq, int batch, int hidden>
void run_bench(std::ofstream& fout, const int i2, const int i3, const int i4,
               const int i5, const int i6) {
    using WholeShape = TileShape<4 * hidden, batch, hidden>;
    using CellShape2 = TileShape<4 * hidden, 2 * batch, hidden>;
    using CellShape3 = TileShape<4 * hidden, 4 * batch, hidden>;
    using CellShape4 = TileShape<4 * hidden, 8 * batch, hidden>;
    using CellShape5 = TileShape<4 * hidden, 16 * batch, hidden>;
    using CellShape6 = TileShape<4 * hidden, 32 * batch, hidden>;

    float elapsed_time = 0.0f;
    elapsed_time +=
        run_dilated_region1<cutlass::half_t, TileShape<2, 2, 1>,
                            TileShape<16 * 4, 32 * 2, 32 * 2>, WholeShape,
                            CellShape2>(depth, seq, batch, hidden, fout);
    if (depth >= 2) {
        elapsed_time +=
            run_dilated_lstm_cell<cutlass::half_t, TileShape<2, 2, 1>,
                                  TileShape<16 * 4, 32 * 2, 32 * 2>,
                                  CellShape2>(depth, seq, batch, hidden, i2,
                                              fout);
    }

    if (depth >= 3) {
        elapsed_time +=
            run_dilated_lstm_cell<cutlass::half_t, TileShape<2, 2, 1>,
                                  TileShape<16 * 4, 32 * 2, 32 * 2>,
                                  CellShape3>(depth, seq, batch, hidden, i3,
                                              fout);
    }

    if (depth >= 4) {
        elapsed_time +=
            run_dilated_lstm_cell<cutlass::half_t, TileShape<2, 2, 1>,
                                  TileShape<16 * 4, 32 * 2, 32 * 2>,
                                  CellShape4>(depth, seq, batch, hidden, i4,
                                              fout);
    }

    if (depth >= 5) {
        elapsed_time +=
            run_dilated_lstm_cell<cutlass::half_t, TileShape<2, 2, 1>,
                                  TileShape<16 * 4, 32 * 2, 32 * 2>,
                                  CellShape5>(depth, seq, batch, hidden, i5,
                                              fout);
    }

    if (depth >= 6) {
        elapsed_time +=
            run_dilated_lstm_cell<cutlass::half_t, TileShape<2, 2, 1>,
                                  TileShape<16 * 4, 32 * 2, 32 * 2>,
                                  CellShape6>(depth, seq, batch, hidden, i6,
                                              fout);
    }

    int input_size = hidden;
    genSeqs(batch, seq, false);
    float cudnn_time =
        bench_cudnn_lstm_half(batch, hidden, seq, depth, input_size);

    std::cout << "depth: " << depth << ", seq_length: " << seq
              << ", batch_size: " << batch << ", hidden_size: " << hidden
              << ", Ours(ms): " << elapsed_time << "ms"
              << ", CuDNN(ms):" << cudnn_time << "ms"
              << ", " << elapsed_time / cudnn_time << std::endl;

    fout << depth << "\t"
         << "[" << seq << ", " << batch << ", " << hidden << "]\t"
         << "[" << 4 * hidden << ", " << batch << ", " << hidden << "]\t"
         << elapsed_time << "\t" << cudnn_time << "\t"
         << elapsed_time / cudnn_time << std::endl;
}

int main(int argc, char* argv[]) {
    assert(argc == 2);
    const char* filename = argv[1];

    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    fout.open(filename, std::ios::out);

    std::cout << "Cute Dilated Stacked Lstm Sim Benchmark:" << std::endl;
    fout << "depth\t[seq_length, batch_size, hidden_size]\t[kTM, kTN, "
            "kTK]\tOurs(ms)\tCuDNN(ms)\tRatio"
         << std::endl;

    // depth, seq, batch, hidden
    run_bench<6, 50, 256, 256>(fout, 25, 13, 7, 4, 2);
    run_bench<6, 50, 256, 512>(fout, 25, 13, 7, 4, 2);
    run_bench<6, 50, 256, 1024>(fout, 25, 13, 7, 4, 2);

    // scale with depth
    run_bench<1, 50, 256, 256>(fout, 25, 13, 7, 4, 2);
    run_bench<2, 50, 256, 256>(fout, 25, 13, 7, 4, 2);
    run_bench<3, 50, 256, 256>(fout, 25, 13, 7, 4, 2);
    run_bench<4, 50, 256, 256>(fout, 25, 13, 7, 4, 2);
    run_bench<5, 50, 256, 256>(fout, 25, 13, 7, 4, 2);
    run_bench<6, 50, 256, 256>(fout, 25, 13, 7, 4, 2);

    run_bench<1, 50, 256, 1024>(fout, 25, 13, 7, 4, 2);
    run_bench<2, 50, 256, 1024>(fout, 25, 13, 7, 4, 2);
    run_bench<3, 50, 256, 1024>(fout, 25, 13, 7, 4, 2);
    run_bench<4, 50, 256, 1024>(fout, 25, 13, 7, 4, 2);
    run_bench<5, 50, 256, 1024>(fout, 25, 13, 7, 4, 2);
    run_bench<6, 50, 256, 1024>(fout, 25, 13, 7, 4, 2);

    // scale with seq
    run_bench<6, 32, 256, 256>(fout, 16, 8, 4, 2, 1);
    run_bench<6, 64, 256, 256>(fout, 32, 16, 8, 4, 2);
    run_bench<6, 128, 256, 256>(fout, 64, 32, 16, 8, 4);

    run_bench<6, 32, 256, 1024>(fout, 16, 8, 4, 2, 1);
    run_bench<6, 64, 256, 1024>(fout, 32, 16, 8, 4, 2);
    run_bench<6, 128, 256, 1024>(fout, 64, 32, 16, 8, 4);

    return 0;
}
