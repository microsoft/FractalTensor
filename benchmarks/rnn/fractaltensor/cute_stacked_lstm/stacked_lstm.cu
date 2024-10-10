// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuDNN/utils.h"
#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/fill.h"
#include "kaleido/core/device/kernels/lstm/stacked_lstm/region1.h"
#include "kaleido/core/device/kernels/lstm/stacked_lstm/region2.h"
#include "kaleido/core/device/kernels/lstm/stacked_lstm/region3.h"
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
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
float bench_stacked_lstm(const int depth, const int seq_length,
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

    const int warm_up = 5;
    const int repeat = 20;

    for (auto i = 0; i < warm_up; ++i) {
        cuda_kernel::StackedLstmRegion1<Element, InstructionShape, ValueMnk,
                                        WarpArragement, CtaTileShape,
                                        WholeShape>(dHsss, dCsss, dXs, dWs, dUs,
                                                    depth, seq_length,
                                                    batch_size, hidden_size);

        cuda_kernel::StackedLstmRegion2<Element, InstructionShape, ValueMnk,
                                        WarpArragement, CtaTileShape,
                                        WholeShape>(dHsss, dCsss, dXs, dWs, dUs,
                                                    depth, seq_length,
                                                    batch_size, hidden_size);

        cuda_kernel::StackedLstmRegion3<Element, InstructionShape, ValueMnk,
                                        WarpArragement, CtaTileShape>(
            dHsss, dCsss, dXs, dWs, dUs, depth, seq_length, batch_size,
            hidden_size);
    }

    float elapsed_time = 0.0f;

    for (auto i = 0; i < repeat; ++i) {
        elapsed_time +=
            cuda_kernel::StackedLstmRegion1<Element, InstructionShape, ValueMnk,
                                            WarpArragement, CtaTileShape,
                                            WholeShape>(
                dHsss, dCsss, dXs, dWs, dUs, depth, seq_length, batch_size,
                hidden_size);

        elapsed_time +=
            cuda_kernel::StackedLstmRegion2<Element, InstructionShape, ValueMnk,
                                            WarpArragement, CtaTileShape,
                                            WholeShape>(
                dHsss, dCsss, dXs, dWs, dUs, depth, seq_length, batch_size,
                hidden_size);

        elapsed_time +=
            cuda_kernel::StackedLstmRegion3<Element, InstructionShape, ValueMnk,
                                            WarpArragement, CtaTileShape>(
                dHsss, dCsss, dXs, dWs, dUs, depth, seq_length, batch_size,
                hidden_size);
    }

    CudaCheck(cudaFree(dWs));
    CudaCheck(cudaFree(dXs));
    CudaCheck(cudaFree(dUs));
    CudaCheck(cudaFree(dCsss));
    CudaCheck(cudaFree(dHsss));

    int input_size = hidden_size;
    genSeqs(batch_size, seq_length, false);
    float cudnn_time = bench_cudnn_lstm_half(batch_size, hidden_size,
                                             seq_length, depth, input_size);

    std::cout << "depth: " << depth << ", seq_length: " << seq_length
              << ", batch_size: " << batch_size
              << ", hidden_size: " << hidden_size
              << ", Ours(ms): " << elapsed_time / repeat << "ms"
              << ", CuDNN(ms):" << cudnn_time << "ms" << std::endl;

    static const int kTM = dim_size<0, CtaTileShape>;
    static const int kTN = dim_size<1, CtaTileShape>;
    static const int kTK = dim_size<2, CtaTileShape>;

    fout << depth << "\t"
         << "[" << seq_length << ", " << batch_size << ", " << hidden_size
         << "]\t[" << kTM << ", " << kTN << ", " << kTK << "]\t"
         << elapsed_time / repeat << "\t" << cudnn_time << std::endl;

    return elapsed_time / repeat;
}

int count = 1;

template <const int depth, const int seq, const int batch, int hidden,
          const int m = 64, const int n = 64, const int k = 64>
void run_bench(std::ofstream& fout) {
    bench_stacked_lstm<cutlass::half_t, TileShape<2, 2, 1>, /*Warp Arrangement*/
                       TileShape<m, n, k>,                  /*CTA Shape*/
                       TileShape<4 * hidden, batch, hidden>>(depth, seq, batch,
                                                             hidden, fout);
    cudaDeviceSynchronize();
}

int main(int argc, char* argv[]) {
    assert(argc == 2);
    const char* filename = argv[1];

    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    fout.open(filename, std::ios::out);

    std::cout << "Cute Stacked Lstm Benchmark" << std::endl;
    fout << "depth\t[seq_length, batch_size, hidden_size]\t[kTM, kTN, "
            "kTK]\tOurs(ms)\tCuDNN(ms)"
         << std::endl;

    // overall performance
    run_bench<32 /*depth*/, 64 /*length*/, 256 /*batch*/, 256 /*hidden*/>(fout);
    run_bench<32, 64, 256, 512>(fout);
    run_bench<32, 64, 256, 1024>(fout);

    // varying depth
    run_bench<1, 64, 256, 256, 32, 32, 64>(fout);
    run_bench<2, 64, 256, 256>(fout);
    run_bench<4, 64, 256, 256>(fout);
    run_bench<8, 64, 256, 256, 64, 64, 64>(fout);
    run_bench<16, 64, 256, 256>(fout);
    run_bench<32, 64, 256, 256>(fout);

    run_bench<1, 64, 256, 1024, 32, 64, 64>(fout);
    run_bench<2, 64, 256, 1024>(fout);
    run_bench<4, 64, 256, 1024>(fout);
    run_bench<8, 64, 256, 1024, 64, 64, 64>(fout);
    run_bench<16, 64, 256, 1024>(fout);
    run_bench<32, 64, 256, 1024>(fout);

    // varying seq_length
    run_bench<32, 32, 256, 256>(fout);
    run_bench<32, 64, 256, 256>(fout);
    run_bench<32, 128, 256, 256>(fout);

    run_bench<32, 32, 256, 1024>(fout);
    run_bench<32, 64, 256, 1024>(fout);
    run_bench<32, 128, 256, 1024>(fout);

    // figure 2
    run_bench<1, 64, 256, 256>(fout);
    run_bench<4, 64, 256, 256>(fout);
    run_bench<8, 64, 256, 256>(fout);
    run_bench<12, 64, 256, 256>(fout);
    run_bench<16, 64, 256, 256>(fout);
    run_bench<20, 64, 256, 256>(fout);
    return 0;
}
