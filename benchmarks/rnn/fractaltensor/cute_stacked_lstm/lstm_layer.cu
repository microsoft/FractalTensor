#include "cuDNN/utils.h"
#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/gemm.h"
#include "kaleido/core/device/kernels/lstm.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tile_shape.h"

#include <cutlass/gemm/device/gemm.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace kaleido::core;

namespace {

float TestCuDNNLSTMHalf(int mini_batch, int hidden_size, int seq_length,
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

template <int batch_size, int hidden_size, int seq_length, int depth,
          typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void bench_cute_lstm_layer() {
    srand(42);

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

    size_t numel = depth * 4 * hidden_size * hidden_size;

    thrust::host_vector<Element> hWs(numel);
    for (int i = 0; i < hWs.size(); ++i)
        hWs[i] = __float2half(rand() / float(RAND_MAX) / 10.);
    thrust::device_vector<Element> dWs = hWs;

    numel = seq_length * batch_size * hidden_size;
    thrust::host_vector<Element> hXs(numel);
    for (int i = 0; i < hXs.size(); ++i)
        hXs[i] = __float2half(rand() / float(RAND_MAX) / 10.);
    thrust::device_vector<Element> dXs = hXs;

    numel = depth * 4 * hidden_size * hidden_size;
    thrust::host_vector<Element> hUs(numel);
    for (int i = 0; i < hUs.size(); ++i)
        hUs[i] = __float2half(rand() / float(RAND_MAX) / 10.);
    thrust::device_vector<Element> dUs = hUs;

    numel = depth * seq_length * batch_size * hidden_size;
    thrust::device_vector<Element> dCsss(numel);
    thrust::fill(dCsss.begin(), dCsss.end(), static_cast<Element>(0.));

    thrust::device_vector<Element> dHsss(numel);
    thrust::fill(dHsss.begin(), dHsss.end(), static_cast<Element>(0.));

    thrust::device_vector<Element> dC_init(numel);
    thrust::fill(dC_init.begin(), dC_init.end(), static_cast<Element>(0.));

    thrust::device_vector<Element> dH_init(numel);
    thrust::fill(dH_init.begin(), dH_init.end(), static_cast<Element>(0.));

    using LSTMLayer =
        cuda_kernel::CuteLSTMLayer<Element, InstructionShape, ValueMnk,
                                   WarpArrangement, CtaTileShape, WholeShape>;
    LSTMLayer lstm_layer;

    const int warm_up = 5;
    for (int i = 0; i < warm_up; ++i) {
        lstm_layer(thrust::raw_pointer_cast(dWs.data()),
                   thrust::raw_pointer_cast(dXs.data()),
                   thrust::raw_pointer_cast(dUs.data()),
                   thrust::raw_pointer_cast(dC_init.data()),
                   thrust::raw_pointer_cast(dH_init.data()),
                   thrust::raw_pointer_cast(dCsss.data()),
                   thrust::raw_pointer_cast(dHsss.data()), seq_length);
    }

    const int iter_counts = 20;
    float time = 0.;
    for (int i = 0; i < iter_counts; ++i) {
        lstm_layer(thrust::raw_pointer_cast(dWs.data()),
                   thrust::raw_pointer_cast(dXs.data()),
                   thrust::raw_pointer_cast(dUs.data()),
                   thrust::raw_pointer_cast(dC_init.data()),
                   thrust::raw_pointer_cast(dH_init.data()),
                   thrust::raw_pointer_cast(dCsss.data()),
                   thrust::raw_pointer_cast(dHsss.data()), seq_length, &time);
    }
    time /= iter_counts;

    std::cout << "[" << batch_size << ", " << hidden_size << ", " << seq_length
              << "]\t[" << kM << ", " << kN << ", " << kK << "]\t[" << kTM
              << ", " << kTN << ", " << kTK << "]\t" << time << "\t";

    int input_size = hidden_size;
    genSeqs(batch_size, seq_length, false);
    float cudnn_time = TestCuDNNLSTMHalf(batch_size, hidden_size, seq_length,
                                         depth, input_size);
    std::cout << cudnn_time << "\t" << time / cudnn_time << std::endl;
}

template <int batch, int hidden, int length>
void run_test() {
    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<1, 1, 1>,    /*Warp Arrangement*/
                          TileShape<16, 32, 32>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();

    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<2, 1, 1>,        /*Warp Arrangement*/
                          TileShape<16 * 4, 32, 32>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();
    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<2, 1, 1>,            /*Warp Arrangement*/
                          TileShape<16 * 4, 32, 32 * 2>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();

    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<2, 1, 1>, /*Warp Arrangement*/
                          TileShape<16 * 4, 32 * 2, 32 * 2>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();
    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<2, 1, 1>, /*Warp Arrangement*/
                          TileShape<16 * 4, 32 * 2, 32 * 4>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();
    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<2, 1, 1>, /*Warp Arrangement*/
                          TileShape<16 * 8, 32 * 2, 32 * 2>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();

    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<2, 2, 1>, /*Warp Arrangement*/
                          TileShape<16 * 4, 32 * 2, 32 * 2>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();
    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<2, 2, 1>,            /*Warp Arrangement*/
                          TileShape<16 * 8, 32, 32 * 4>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();
    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<2, 2, 1>, /*Warp Arrangement*/
                          TileShape<16 * 8, 32 * 2, 32 * 4>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();

    bench_cute_lstm_layer<batch, hidden, length, 1, cutlass::half_t,
                          TileShape<4, 2, 1>, /*Warp Arrangement*/
                          TileShape<16 * 8, 32 * 2, 32 * 2>, /*CTA Shape*/
                          TileShape<4 * hidden, batch, hidden>>();
}
}  // namespace

int main() {
    std::cout << "[batch, hidden, length]\tGEMM Shape\tCTA "
                 "Shape\tOurs(ms)\tCuDNN(ms)\tRatio to CuDNN"
              << std::endl;

    run_test<256 /*batch*/, 128 /*hidden*/, 64 /*length*/>();
    run_test<256 /*batch*/, 256 /*hidden*/, 64 /*length*/>();
    run_test<256 /*batch*/, 512 /*hidden*/, 64 /*length*/>();
    run_test<256 /*batch*/, 1024 /*hidden*/, 64 /*length*/>();
    run_test<256 /*batch*/, 2048 /*hidden*/, 64 /*length*/>();
}
