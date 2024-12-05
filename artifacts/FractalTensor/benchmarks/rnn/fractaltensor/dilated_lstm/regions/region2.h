#pragma once

#include "utils.h"

namespace kaleido {
namespace core {

void Region2DepthFirstReuse(const GPUContext& context, Tensor& hsss,
                            Tensor& csss, const Tensor& xss, const Tensor& ws,
                            const Tensor& us,
                            std::shared_ptr<Allocator> allocator,
                            std::vector<float>& time_info) {
    if (ws.dim_size(0) == 1) return;

    std::vector<float> compute(2, 0.);

    /* Loop bounds:
      1 <= i < D  // i -> depth
      0 <= j < L  // j -> length
      0 <= k < N  // k -> batch
      */

    int64_t lb0 = 1;
    int64_t ub0 = ws.dim_size(0);

    int64_t lb1 = 0;
    int64_t ub1 = xss.dim_size(0);

    int64_t lb2 = 0;
    int64_t ub2 = xss.dim_size(1);

    int64_t link_len = 2;

    int64_t stride_i =
        hsss.dim_size(-3) * hsss.dim_size(-2) * hsss.dim_size(-1);
    int64_t stride_j = hsss.dim_size(-2) * hsss.dim_size(-1);

    for (int64_t i = lb0; i < ub0; ++i) {
        // Allocate
        Tensor tmp1({link_len * xss.dim_size(1), ws.dim_size(-1)}, allocator);
        Tensor tmp2({link_len * xss.dim_size(1), ws.dim_size(-1)}, allocator);
        FillZeros(tmp1);
        FillZeros(tmp2);

        Tensor tmp3({link_len * xss.dim_size(1), ws.dim_size(-1)}, allocator);
        Tensor tmp4({link_len * xss.dim_size(1), ws.dim_size(-1)}, allocator);
        FillZeros(tmp3);
        FillZeros(tmp4);

        Tensor out1({link_len * csss.dim_size(-2), csss.dim_size(-1)}, nullptr);
        out1.CreateFrom<float>(csss, i * stride_i);

        Tensor out2({link_len * hsss.dim_size(-2), hsss.dim_size(-1)}, nullptr);
        out2.CreateFrom<float>(hsss, i * stride_i);

        Tensor w({ws.dim_size(1), ws.dim_size(2)}, nullptr);
        w.CreateFrom<float>(ws, i * ws.dim_size(1) * ws.dim_size(2));
        Tensor u({us.dim_size(1), us.dim_size(2)}, nullptr);
        u.CreateFrom<float>(us, i * us.dim_size(1) * us.dim_size(2));

        Tensor x({link_len * xss.dim_size(1), hsss.dim_size(-1)}, nullptr);
        Tensor h({link_len * xss.dim_size(1), hsss.dim_size(-1)}, nullptr);
        Tensor c({link_len * xss.dim_size(1), csss.dim_size(-1)}, nullptr);

        x.CreateFrom<float>(hsss, (i - 1) * stride_i);
        Tensor h0({link_len * xss.dim_size(1), hsss.dim_size(-1)}, allocator);
        FillZeros(h0);
        Tensor c0({link_len * xss.dim_size(1), hsss.dim_size(-1)}, allocator);
        FillZeros(c0);

        auto times = LstmCell(context, x, w, c0, h0, u, out1, out2, tmp1, tmp2,
                              tmp3, tmp4);

        compute[0] += times[0];
        compute[1] += times[1];

        for (int64_t j = lb1 + link_len; j < ub1; j += link_len) {
            out1.CreateFrom<float>(csss, i * stride_i + j * stride_j);
            out2.CreateFrom<float>(hsss, i * stride_i + j * stride_j);

            x.CreateFrom<float>(hsss, (i - 1) * stride_i + j * stride_j);
            h.CreateFrom<float>(hsss, i * stride_i + (j - link_len) * stride_j);
            c.CreateFrom<float>(csss, i * stride_i + (j - link_len) * stride_j);

            auto times = LstmCell(context, x, w, c, h, u, out1, out2, tmp1,
                                  tmp2, tmp3, tmp4);

            compute[0] += times[0];
            compute[1] += times[1];
        }
        link_len = 2 * link_len;
    }

    time_info[0] += 0.;
    time_info[1] += compute[0];
    time_info[2] += compute[1];
    time_info[3] += 0.;
}

}  // namespace core
}  // namespace kaleido
