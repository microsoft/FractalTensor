#pragma once

#include "kaleido/core/device/cuda_utils.h"

namespace kaleido::core::cuda_kernel {

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
float DilatedLstmRegion2(Element* csss, Element* hsss, const Element* xss,
                         const Element* ws, const Element* us, const int depth,
                         const int seq_length, const int batch_size,
                         const int hidden_size) {
    /* Loop bounds:
    1 <= i < D  // i -> depth
    0 <= j < L  // j -> length
    0 <= k < N  // k -> batch
    */
    float elapsed_time = 0.0;

    int64_t lb0 = 1;
    int64_t ub0 = depth;

    int64_t lb1 = 0;
    int64_t ub1 = seq_length;

    int64_t lb2 = 0;
    int64_t ub2 = batch_size;

    int64_t link_len = 2;

    int64_t stride_i = seq_length * batch_size * hidden_size;
    int64_t stride_j = batch_size * hidden_size;

    int call_lstm_cell_counts = 0;
    for (int64_t i = lb0; i < ub0; ++i) {
        int iter_lstm_cell_counts = 0;
        Element* c_out = csss + i * stride_i;
        Element* h_out = hsss + i * stride_i;
        const Element* w = ws + i * 4 * hidden_size * hidden_size;
        const Element* u = us + i * 4 * hidden_size * hidden_size;
        const Element* x = hsss + (i - 1) * stride_i;

        Element* init;
        CudaCheck(cudaMalloc(
            &init, link_len * batch_size * hidden_size * sizeof(Element)));

        using LstmCell = cuda_kernel::CuteDynamicLstmCell<
            Element, InstructionShape, ValueMnk, WarpArrangement, CtaTileShape>;
        LstmCell lstm_cell;

        int m = 4 * hidden_size;
        int n = link_len * batch_size;
        int k = hidden_size;

        call_lstm_cell_counts++;
        iter_lstm_cell_counts++;
        elapsed_time += lstm_cell(w, x, u, init, init, c_out, h_out, m, n, k);

        CudaCheck(cudaFree(init));

        Element *h, *c;

        for (int64_t j = lb1 + link_len; j < ub1; j += link_len) {
            c_out = csss + i * stride_i + j * stride_j;
            h_out = hsss + i * stride_i + j * stride_j;
            x = hsss + (i - 1) * stride_i + j * stride_j;
            h = hsss + i * stride_i + (j - link_len) * stride_j;
            c = csss + i * stride_i + (j - link_len) * stride_j;

            call_lstm_cell_counts++;
            iter_lstm_cell_counts++;
            elapsed_time += lstm_cell(w, x, u, c, h, c_out, h_out, m, n, k);
        }

        link_len = 2 * link_len;
    }
    return elapsed_time;
};
}  // namespace kaleido::core::cuda_kernel
