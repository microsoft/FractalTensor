#pragma once
#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/kernels/lstm.h"
#include "kaleido/core/device/kernels/lstm_ref.h"
#include "kaleido/core/device/kernels/scatter_nd.h"
#include "kaleido/core/tile_shape.h"

#include <algorithm>
#include <vector>

namespace kaleido::core::cuda_kernel {

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
float StackedLstmRegion3(Element* hsss, Element* csss, const Element* xss,
                         const Element* ws, const Element* us, const int depth,
                         const int seq_length, const int batch_size,
                         const int hidden_size) {
    // us: [depth, 4 * hidden_size, hidden_size]
    // ws: [depth, 4 * hidden_size, hidden_size]
    // xss: [seq_length, batch_size, hidden_size]
    // hsss: [depth, seq_length, batch_size, hidden_size]
    // csss: [depth, seq_length, batch_size, hidden_size]

    CudaTimer timer;
    float elapsed_time = 0.0f;

    const int k_depth_1 = depth;
    if (k_depth_1 == 1) return elapsed_time;

    const int k_depth_2 = seq_length;
    const int k_depth_3 = batch_size;

    const int k_vec_size = hidden_size;

    auto cal_offset = [&](const std::vector<int>& ids) {
        auto offset = (ids[0] * k_depth_2 + ids[1]) * k_depth_3 * k_vec_size;
        return offset;
    };

    const int lb3 = 2;
    const int ub3 = seq_length + depth - 1;

    int lb1 = 0;
    int ub1 = batch_size;

    for (int m = lb3; m < ub3; ++m) {
        // Begion Prologue of a Control Region
        const int lb2 = std::max(1, m - k_depth_2 + 1);
        const int ub2 = std::min(k_depth_1, m);
        const int length_depth_2 = ub2 - lb2;

        int j0 = std::max(1, m - k_depth_2 + 1);

        std::vector<int> start_xs{j0 - 1, m - j0, 0};
        std::vector<int> start_hs{j0, m - j0 - 1, 0};

        int stride = k_depth_2 - 1;
        int stried_span = (length_depth_2 - 1) * stride + 1;

        const Element* xs = hsss + cal_offset(start_xs);
        const Element* w = ws + lb2 * 4 * hidden_size * hidden_size;
        const Element* u = us + lb2 * 4 * hidden_size * hidden_size;
        Element* cs = csss + cal_offset(start_hs);
        Element* hs = hsss + cal_offset(start_hs);

        Element* c_out;
        Element* h_out;
        cudaMalloc((void**)&c_out,
                   length_depth_2 * k_depth_3 * hidden_size * sizeof(Element));
        cudaMalloc((void**)&h_out,
                   length_depth_2 * k_depth_3 * hidden_size * sizeof(Element));

        const int kM = 4 * hidden_size;
        const int kN = batch_size;
        const int kK = hidden_size;
        const int batch = length_depth_2;

        stride *= (hidden_size * batch_size);

        int stride_a = stride;
        int stride_b = 4 * hidden_size * hidden_size;
        int stride_c = k_depth_3 * hidden_size;

#ifdef DEBUG
        std::cout << "kM: " << kM << " kN: " << kN << " kK: " << kK
                  << " batch: " << batch << std::endl;
        std::cout << "stride_a: " << stride_a << " stride_b: " << stride_b
                  << " stride_c: " << stride_c << std::endl;
#endif

        using FusedBMMLSTMCell = cuda_kernel::CuteFusedBMMLSTMCell<
            Element, InstructionShape, ValueMnk, WarpArrangement, CtaTileShape>;

        FusedBMMLSTMCell fused_bmm_lstm_cell;

        elapsed_time += fused_bmm_lstm_cell(
            w, xs, u, cs, hs, c_out, h_out, depth, seq_length, batch_size,
            hidden_size, stride_a, stride_b, stride_c, kM, kN, kK, batch);

        // Scatter
        std::vector<int64_t> data_dims = {depth, seq_length, batch_size,
                                          hidden_size};
        std::vector<int64_t> indices_dims = {length_depth_2 * k_depth_3, 3};

        size_t slice_size = hidden_size;

        // index over the first 3 dimensionalities, the remaining dimensions are
        // to slice over.
        size_t k = 3;
        size_t remain_size = length_depth_2 * k_depth_3;
        // cpu
        int64_t* host_indices =
            (int64_t*)malloc(remain_size * k * sizeof(int64_t));

        int count = 0;
        for (int n = lb2; n < ub2; ++n) {
            for (int p = lb1; p < ub1; ++p) {
                host_indices[count * k] = n;
                host_indices[count * k + 1] = m - n;
                host_indices[count * k + 2] = p;
                count++;
            }
        }

        int64_t* device_indices;
        cudaMalloc((void**)&device_indices, remain_size * k * sizeof(int64_t));
        cudaMemcpy(device_indices, host_indices,
                   remain_size * k * sizeof(int64_t), cudaMemcpyHostToDevice);

        cudaFree(device_indices);
        cudaFree(c_out);
        cudaFree(h_out);
    }

    return elapsed_time;
}

}  // namespace kaleido::core::cuda_kernel
