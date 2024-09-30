#pragma once

#include "utils.h"

namespace kaleido {
namespace core {

void Region3DepthFirst(const GPUContext& context, Tensor& hsss,
                       const Tensor& xss, const Tensor& yss, const Tensor& ws,
                       const Tensor& us, const Tensor& bs,
                       std::shared_ptr<Allocator> allocator,
                       std::vector<float>& time_info) {
    if (yss.dim_size(0) <= 1) return;

    ops::PrintOp<GPUContext, CUDAPlace, float> printer;

    std::vector<float> compute(2, 0.);
    float total_gather = 0.f;
    float total_scatter = 0.f;

    /* ============ Region 3 is controlled by s single control
     ================ (0, 0, >0, *)
     [[1, 0, 0, 0]
      [0, 1, 0, 0]
      [0, 0, 1, 0]
      [0, 0, 0, 1]]
      m = 0           <--> d = 0          // d -> depth
      n = 0           <--> i = 0          // i -> src_length
      p = j           <--> j = p          // j -> trg_length
      q = k           <--> k = q          // k -> batch

      Loop bounds:
      1 <= p < TL
      0 <= q < N

      Loop invariant inputs:
      x_t
      w, u, b
      ========================================================================*/

    int64_t depth_size = ws.dim_size(0);
    int64_t src_length = hsss.dim_size(1);
    int64_t trg_length = hsss.dim_size(2);
    int64_t batch_size = xss.dim_size(-2);
    int64_t hidden_size = ws.dim_size(-1);

    Tensor x_t({1, batch_size, hidden_size}, nullptr);
    Tensor y_t({1, batch_size, hidden_size}, nullptr);
    x_t.CreateFrom<float>(xss, 0);

    int kGridDim = 2;
    Tensor w({1, hidden_size, hidden_size}, nullptr);
    Tensor u({1, kGridDim * hidden_size, hidden_size}, nullptr);
    Tensor b({1, batch_size, hidden_size}, nullptr);
    w.CreateFrom<float>(ws, 0);
    u.CreateFrom<float>(us, 0);
    b.CreateFrom<float>(bs, 0);

    Tensor tmp1({1, batch_size, hidden_size}, allocator);
    Tensor tmp2({1, batch_size, hidden_size}, allocator);
    Tensor tmp3({1, batch_size, hidden_size}, allocator);
    Tensor tmp4({1, batch_size, hidden_size}, allocator);
    FillZeros(tmp1);
    FillZeros(tmp2);
    FillZeros(tmp3);
    FillZeros(tmp4);

    int stride_h_y = hsss.dim_size(-3) * hsss.dim_size(-2) * hsss.dim_size(-1);
    int stride_h_x_y = hsss.dim_size(-2) * hsss.dim_size(-1);

    Tensor h_x({1, batch_size, hidden_size}, nullptr);
    Tensor h_y({1, batch_size, hidden_size}, nullptr);

    for (size_t p = 1; p < yss.dim_size(0); ++p) {
        y_t.CreateFrom<float>(yss, p * yss.dim_size(-2) * yss.dim_size(-1));

        Tensor state_x({1, batch_size, hidden_size}, allocator);
        Tensor state_y({1, batch_size, hidden_size}, nullptr);
        state_y.CreateFrom<float>(hsss, (p - 1) * stride_h_y + stride_h_x_y);
        FillZeros(state_x);
        Tensor state({1, batch_size, 2 * hidden_size}, allocator);
        FillZeros(state);

        std::vector<Tensor> tensors;
        tensors.emplace_back(state_x);
        tensors.emplace_back(state_y);
        ops::ConcatOp<GPUContext, CUDAPlace, float> cat;
        cat(context, tensors, state, 2);

        h_x.CreateFrom<float>(hsss, p * stride_h_y);
        h_y.CreateFrom<float>(hsss, p * stride_h_y + stride_h_x_y);

        auto times = VanillaRNNCellBMM(context, x_t, y_t, state, w, u, b, tmp1,
                                       tmp2, tmp3, tmp4, h_x, h_y, batch_size,
                                       hidden_size, hidden_size, 1);
        compute[0] += times[0];
        compute[1] += times[1];
    }

#ifdef DEBUG_ON
    std::cout << "region3: hsss:" << std::endl;
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
