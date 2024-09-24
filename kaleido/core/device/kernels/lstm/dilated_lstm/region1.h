#pragma once

#include "kaleido/core/device/kernels/lstm.h"
#include "kaleido/core/tensor.h"

namespace kaleido::core::cuda_kernel {

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
float DilatedLstmRegion1(Element* csss, Element* hsss, const Element* xss,
                         const Element* ws, const Element* us,
                         const Element* init, int seq_length) {
    float elapsed_time = 0.0;

    const Element* x = xss;
    const Element* w = ws;
    const Element* u = us;

    cuda_kernel::CuteLSTMLayer<Element, InstructionShape, ValueMnk,
                               WarpArrangement, CtaTileShape, WholeShape>
        cute_lstm_layer;

    elapsed_time +=
        cute_lstm_layer(w, x, u, init, init, csss, hsss, seq_length);

    return elapsed_time;
}
}  // namespace kaleido::core::cuda_kernel
