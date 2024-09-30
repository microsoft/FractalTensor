#pragma once

#include "utils.h"

namespace kaleido {
namespace core {

void Region5DepthFirst(const GPUContext& context, Tensor& hsss,
                       const Tensor& xss, const Tensor& yss, const Tensor& ws,
                       const Tensor& us, const Tensor& bs,
                       std::shared_ptr<Allocator> allocator,
                       std::vector<float>& time_info) {
    if (ws.dim_size(0) <= 1) return;

    ops::PrintOp<GPUContext, CUDAPlace, float> printer;

    std::vector<float> compute(2, 0.);
    float total_gather = 0.f;
    float total_scatter = 0.f;

    /* ============ Region 5 is controlled by s single control
     ================
     (>0, 0, 0, *)
     [[1, 0, 0, 0]
      [0, 1, 0, 0]
      [0, 0, 1, 0]
      [0, 0, 0, 1]]
      m = d           <--> d = m          // d -> depth
      n = 0           <--> i = 0          // i -> src_length
      p = 0           <--> j = 0          // j -> trg_length
      q = k           <--> k = q          // k -> batch

      Loop bounds:
      1 <= m < D
      0 <= q < N

      Loop invariant inputs:
      ========================================================================*/

    int64_t depth_size = ws.dim_size(0);
    int64_t src_length = hsss.dim_size(1);
    int64_t trg_length = hsss.dim_size(2);
    int64_t batch_size = xss.dim_size(-2);
    int64_t hidden_size = ws.dim_size(-1);

    Tensor x_t({1, batch_size, hidden_size}, nullptr);
    Tensor y_t({1, batch_size, hidden_size}, nullptr);

    int kGridDim = 2;
    Tensor w({1, hidden_size, hidden_size}, nullptr);
    Tensor u({1, kGridDim * hidden_size, hidden_size}, nullptr);
    Tensor b({1, batch_size, hidden_size}, nullptr);

    Tensor tmp1({1, batch_size, hidden_size}, allocator);
    Tensor tmp2({1, batch_size, hidden_size}, allocator);
    Tensor tmp3({1, batch_size, hidden_size}, allocator);
    Tensor tmp4({1, batch_size, hidden_size}, allocator);
    FillZeros(tmp1);
    FillZeros(tmp2);
    FillZeros(tmp3);
    FillZeros(tmp4);

    int stride_h_d = hsss.dim_size(-5) * hsss.dim_size(-4) * hsss.dim_size(-3) *
                     hsss.dim_size(-2) * hsss.dim_size(-1);
    int stride_h_x_y = hsss.dim_size(-2) * hsss.dim_size(-1);

    Tensor h_x({1, batch_size, hidden_size}, nullptr);
    Tensor h_y({1, batch_size, hidden_size}, nullptr);

    for (size_t m = 1; m < ws.dim_size(0); ++m) {
        x_t.CreateFrom<float>(hsss, (m - 1) * stride_h_d);
        y_t.CreateFrom<float>(hsss, (m - 1) * stride_h_d + stride_h_x_y);

        Tensor state_x({1, batch_size, hidden_size}, allocator);
        Tensor state_y({1, batch_size, hidden_size}, allocator);
        FillZeros(state_x);
        FillZeros(state_y);
        Tensor state({1, batch_size, 2 * hidden_size}, allocator);
        FillZeros(state);

        std::vector<Tensor> tensors;
        tensors.emplace_back(state_x);
        tensors.emplace_back(state_y);
        ops::ConcatOp<GPUContext, CUDAPlace, float> cat;
        cat(context, tensors, state, 2);

        w.CreateFrom<float>(ws, m * ws.dim_size(-2) * ws.dim_size(-1));
        u.CreateFrom<float>(us, m * us.dim_size(-2) * us.dim_size(-1));
        b.CreateFrom<float>(bs, m * bs.dim_size(-2) * bs.dim_size(-1));

        h_x.CreateFrom<float>(hsss, m * stride_h_d);
        h_y.CreateFrom<float>(hsss, m * stride_h_d + stride_h_x_y);
        auto times = VanillaRNNCellBMM(context, x_t, y_t, state, w, u, b, tmp1,
                                       tmp2, tmp3, tmp4, h_x, h_y, batch_size,
                                       hidden_size, hidden_size, 1);
        compute[0] += times[0];
        compute[1] += times[1];
    }

#ifdef DEBUG_ON
    std::cout << "region5: hsss:" << std::endl;
    std::cout << hsss.DebugString() << std::endl;
    std::cout << printer(hsss) << std::endl;
#endif

    time_info[0] += total_gather;
    time_info[1] += compute[0];
    time_info[2] += compute[1];
    time_info[3] += total_scatter;
}

}  // namespace core
}  // namespace kaleido
