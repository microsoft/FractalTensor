#pragma once
#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/kernels/lstm.h"

namespace kaleido::core::cuda_kernel {
template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArragement, typename CtaTileShape, typename WholeShape>
float StackedLstmRegion1(Element* hsss, Element* csss, const Element* xss,
                         const Element* ws, const Element* us, const int depth,
                         const int seq_length, const int batch_size,
                         const int hidden_size) {
    CudaTimer timer;
    const Element* x = xss;
    Element* init;
    cudaMalloc((void**)&init, sizeof(Element) * hidden_size * batch_size);
    // Fill zero
    cudaMemset(reinterpret_cast<void*>(init), 0,
               sizeof(Element) * hidden_size * batch_size);

    const Element* c_init = init;
    const Element* h_init = init;
    const Element* w = ws;
    const Element* u = us;
    Element* css = csss;
    Element* hss = hsss;

    // TODO: NotFused version.
    using CuteFusedLSTMLayer =
        cuda_kernel::CuteLSTMLayer<Element, InstructionShape, ValueMnk,
                                   WarpArragement, CtaTileShape, WholeShape>;

    CuteFusedLSTMLayer cute_fused_lstm_layer;

    float time =
        cute_fused_lstm_layer(w, x, u, c_init, h_init, css, hss, seq_length);

    CudaCheck(cudaFree(init));

    return time;
}
}  // namespace kaleido::core::cuda_kernel
