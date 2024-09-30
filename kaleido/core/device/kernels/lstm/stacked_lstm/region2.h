#pragma once
#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/kernels/lstm.h"
#include "kaleido/core/device/kernels/lstm_ref.h"
#include "kaleido/core/tile_shape.h"

namespace kaleido::core::cuda_kernel {

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArragement, typename CtaTileShape, typename WholeShape>
float StackedLstmRegion2(Element* hsss, Element* csss, const Element* xss,
                         const Element* ws, const Element* us, const int depth,
                         const int seq_length, const int batch_size,
                         const int hidden_size) {
    if (depth == 1) return 0.;

    CudaTimer timer;
    float elapsed_time = 0;

    int stride = seq_length * batch_size * hidden_size;

    const Element* x = hsss;
    const Element* w = ws + 4 * hidden_size * hidden_size;
    const Element* u = us + 4 * hidden_size * hidden_size;
    const Element* c_init = csss;
    const Element* h_init = hsss;

    Element* c = csss + stride;
    Element* h = hsss + stride;

    int m = hidden_size;
    int n = batch_size;
    int k = hidden_size;

    Element* o;
    CudaCheck(cudaMalloc((void**)&o, 4 * m * n * sizeof(Element)));

    using Region2CuteLSTMCell =
        CuteLSTMCell<Element, InstructionShape, ValueMnk, WarpArragement,
                     CtaTileShape, WholeShape>;
    Region2CuteLSTMCell cute_lstm_cell;

    elapsed_time += cute_lstm_cell(w, x, u, c_init, h_init, c, h, o);

    // w: [4 * hidden_size, hidden_size]
    // xs: [hidden_size, seq_length * batch_size]
    // u: [4 * hidden_size, hidden_size]
    // hs: [hidden_size, seq_length * batch_size]

    for (int i = 2; i < depth; ++i) {
        // LSTM
        const Element* c_1 = csss + (i - 1) * stride;
        const Element* h_1 = hsss + (i - 1) * stride;
        c = csss + i * stride;
        h = hsss + i * stride;
        x = hsss + (i - 1) * stride;
        w = ws + i * 4 * hidden_size * hidden_size;
        u = us + i * 4 * hidden_size * hidden_size;

        elapsed_time += cute_lstm_cell(w, x, u, c_1, h_1, c, h, o);
    }

    CudaCheck(cudaFree(o));

    return elapsed_time;
}

template <typename Element>
void ReferenceRegion2LSTM(cublasHandle_t handle, Element* hsss, Element* csss,
                          const Element* xss, const Element* ws,
                          const Element* us, int seq_length, int batch_size,
                          int hidden_size, int depth) {
    int stride = seq_length * batch_size * hidden_size;
    const Element* x = hsss;
    const Element* w = ws + 4 * hidden_size * hidden_size;
    const Element* u = us + 4 * hidden_size * hidden_size;
    const Element* c_init = csss;
    const Element* h_init = hsss;

    int m = hidden_size;
    int n = batch_size * seq_length;
    int k = hidden_size;

    std::cout << "m: " << m << " n: " << n << " k: " << k << std::endl;

    Element* o;
    cudaMalloc((void**)&o,
               4 * hidden_size * batch_size * seq_length * sizeof(Element));

    Element* c = csss + stride;
    Element* h = hsss + stride;

    cuda_kernel::ReferenceLSTMCell<Element> lstm_cell_reference;
    lstm_cell_reference(handle, w, x, u, c_init, h_init, c, h, o, m, n, k);

    for (int i = 2; i < depth; ++i) {
        // LSTM
        const Element* c_1 = csss + (i - 1) * stride;
        const Element* h_1 = hsss + (i - 1) * stride;
        c = csss + i * stride;
        h = hsss + i * stride;
        x = hsss + (i - 1) * stride;
        w = ws + i * 4 * hidden_size * hidden_size;
        u = us + i * 4 * hidden_size * hidden_size;

        // Act(W@xs, U@hs)
        lstm_cell_reference(handle, w, x, u, c_1, h_1, c, h, o, m, n, k);
    }
}

}  // namespace kaleido::core::cuda_kernel
