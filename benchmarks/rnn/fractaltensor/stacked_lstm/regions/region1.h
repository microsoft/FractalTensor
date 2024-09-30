#pragma once

#include "utils.h"

#include <time.h>

namespace kaleido {
namespace core {

void Region1DepthFirst(const GPUContext& context, Tensor& hsss, Tensor& csss,
                       const Tensor& xss, const Tensor& ws, const Tensor& us,
                       std::shared_ptr<Allocator> allocator,
                       std::vector<float>& time_info) {
    // =============== Begin Prologue of a Control Region
    // ======================= Region 0 is controlled by a single
    // control varialbe: 0<= i0 < batch_size. Region 0 is kept
    // unchanged. (0, *, 0) --> (*, 0) (i0, i2) access `xss` in parallel
    // i2 = 0 access `ws` and `us`
    // ==========================================================================

    float elapsed_gather = 0.f;
    float total_gather = 0.f;

    // (batch_size * seq_length, hidden_size)
    Tensor x({xss.dim_size(-2) * xss.dim_size(-3), xss.dim_size(-1)}, nullptr);
    x.CreateFrom<float>(xss, 0);

    // (batch_size, hidden_size)
    Tensor init({xss.dim_size(-2), xss.dim_size(-1)}, allocator);
    total_gather += FillZeros(init);

    // share the underlying memory. No real copy.
    // (hidden_size, 4 * hidden_size)
    Tensor w({ws.dim_size(-2), ws.dim_size(-1)}, nullptr);
    w.CreateFrom<float>(ws, 0);

    // share the underlying memory. No real copy.
    // (hidden_size, 4 * hidden_size)
    Tensor u({us.dim_size(-2), us.dim_size(-1)}, nullptr);
    u.CreateFrom<float>(us, 0);

    // TODO(ying): shape of cell output is automatically inferred in
    // shape inference.
    // (seq_length, batch_size, hidden_size)
    Tensor out1({csss.dim_size(-3), csss.dim_size(-2), csss.dim_size(-1)},
                nullptr);
    out1.CreateFrom<float>(csss, 0);

    // (seq_length, batch_size, hidden_size)
    Tensor out2({hsss.dim_size(-3), hsss.dim_size(-2), hsss.dim_size(-1)},
                nullptr);
    out2.CreateFrom<float>(hsss, 0);

    // tmp1 = xs @ w, input-to-hidden projection for the entire
    // sequence.
    // (batch_size * seq_length, hidden_size)
    Tensor tmp1({xss.dim_size(0) * xss.dim_size(1), ws.dim_size(-1)},
                allocator);
    // tmp2 = h @ u  // hidden-to-hidden for a single time step
    // (seq_length, batch_size, hidden_size)
    Tensor tmp2({hsss.dim_size(-3), hsss.dim_size(-2), us.dim_size(-1)},
                allocator);
    FillZeros(tmp1);
    FillZeros(tmp2);

    // tmp3 = tmp1 + tmp2
    Tensor tmp3({hsss.dim_size(-3), hsss.dim_size(-2), ws.dim_size(-1)},
                allocator);
    FillZeros(tmp3);

    // tmp4[0:3] = sigmod(tmp3[0:3]), tmp4[-1] = tanh(tmp3[-1])
    Tensor tmp4({hsss.dim_size(-3), hsss.dim_size(-2), ws.dim_size(-1)},
                allocator);
    FillZeros(tmp4);

    // =============== End Prologue of a Control Region
    // =================
    std::vector<float> compute = LstmLayer(context, x, w, init, init, u, out1,
                                           out2, tmp1, tmp2, tmp3, tmp4);

    time_info[0] += total_gather;
    time_info[1] += compute[0];
    time_info[2] += compute[1];
    time_info[3] += 0.;
}

}  // namespace core
}  // namespace kaleido
