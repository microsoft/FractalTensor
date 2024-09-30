#pragma once

#include "kaleido/core/device/cuda_timer.h"
#include "utils.h"

namespace kaleido {
namespace core {

/* Region 2: (>0. 0, *)
 * 1<= i0 < depth and 0 <= i1 < batch_size
 */
void Region2DepthFirst(const GPUContext& context, Tensor& hsss, Tensor& csss,
                       const Tensor& xss, const Tensor& ws, const Tensor& us,
                       std::shared_ptr<Allocator> allocator,
                       std::vector<float>& time_info) {
    if (ws.dim_size(0) == 1) return;

    float elapsed_gather = 0.f;
    float total_gather = 0.f;

    cudaEvent_t start_gather, stop_gather;
    cudaEventCreate(&start_gather);
    cudaEventCreate(&stop_gather);

    Tensor h({xss.dim_size(1), hsss.dim_size(-1)}, allocator);
    cudaEventRecord(start_gather, 0);
    FillZeros(h);
    GpuElapse(start_gather, stop_gather, elapsed_gather, total_gather);

    Tensor tmp1({ws.dim_size(0) - 1, xss.dim_size(1), ws.dim_size(-1)},
                allocator);
    Tensor tmp2({ws.dim_size(0) - 1, xss.dim_size(1), ws.dim_size(-1)},
                allocator);
    FillZeros(tmp1);
    FillZeros(tmp2);

    Tensor tmp3({ws.dim_size(0) - 1, xss.dim_size(1), ws.dim_size(-1)},
                allocator);
    Tensor tmp4({ws.dim_size(0) - 1, xss.dim_size(1), ws.dim_size(-1)},
                allocator);
    FillZeros(tmp3);
    FillZeros(tmp4);

    // =============== End Prologue of a Control Region
    // =================

    // seq_length * batch_size * hidden_size
    int64_t stride = hsss.dim_size(-3) * hsss.dim_size(-2) * hsss.dim_size(-1);

    Tensor out1({csss.dim_size(-2), csss.dim_size(-1)}, nullptr);
    out1.CreateFrom<float>(csss, stride);

    Tensor out2({hsss.dim_size(-2), hsss.dim_size(-1)}, nullptr);
    out2.CreateFrom<float>(hsss, stride);

    // 4 * hidden_size, hidden_size
    Tensor w({ws.dim_size(-2), ws.dim_size(-1)}, nullptr);
    w.CreateFrom<float>(ws, w.numel());

    // 4 * hidden_size, hidden_size
    Tensor u({us.dim_size(-2), us.dim_size(-1)}, nullptr);
    u.CreateFrom<float>(us, u.numel());

    Tensor tmp1_({xss.dim_size(1), ws.dim_size(-1)}, nullptr);
    Tensor tmp2_({xss.dim_size(1), ws.dim_size(-1)}, nullptr);
    tmp1_.CreateFrom<float>(tmp1, 0);
    tmp2_.CreateFrom<float>(tmp2, 0);

    Tensor tmp3_({xss.dim_size(1), ws.dim_size(-1)}, nullptr);
    Tensor tmp4_({xss.dim_size(1), ws.dim_size(-1)}, nullptr);
    tmp3_.CreateFrom<float>(tmp3, 0);
    tmp4_.CreateFrom<float>(tmp4, 0);

    // batch_size * hidden_size
    Tensor x({xss.dim_size(1), hsss.dim_size(-1)}, nullptr);
    x.CreateFrom<float>(hsss, 0);

    // x, w, c, h, u
    std::vector<float> compute = LstmCell(context, x, w, h, h, u, out1, out2,
                                          tmp1_, tmp2_, tmp3_, tmp4_);

    for (size_t i = 2; i < ws.dim_size(0); ++i) {
        // seq_length * batch_size * hidden_size
        out1.CreateFrom<float>(hsss, i * stride);
        out2.CreateFrom<float>(csss, i * stride);

        // seq_length * batch_size * hidden_size
        x.CreateFrom<float>(hsss, (i - 1) * stride);
        // 4 * hidden_size * hidden_size
        w.CreateFrom<float>(ws, i * w.numel());
        // 4 * hidden_size * hidden_size
        u.CreateFrom<float>(us, i * u.numel());

        tmp1_.CreateFrom<float>(tmp1, (i - 1) * tmp1_.numel());
        tmp2_.CreateFrom<float>(tmp2, (i - 1) * tmp2_.numel());
        tmp3_.CreateFrom<float>(tmp3, (i - 1) * tmp3_.numel());
        tmp4_.CreateFrom<float>(tmp4, (i - 1) * tmp4_.numel());

        auto times = LstmCell(context, x, w, h, h, u, out1, out2, tmp1_, tmp2_,
                              tmp3_, tmp4_);

        compute[0] += times[0];
        compute[1] += times[1];
    }

    time_info[0] += total_gather;  // gather
    time_info[1] += compute[0];    // gemm
    time_info[2] += compute[1];    // element-wise
    time_info[3] += 0.;            // scatter
}

}  // namespace core
}  // namespace kaleido
