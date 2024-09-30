#pragma once

#include "utils.h"

namespace kaleido {
namespace core {

/* Region 3: (>0, >0, *)
 *
 * [
 *   [1, 1, 0]
 *   [1, 0, 0]
 *   [0, 0, 1]
 * ]
 *
 *  m = i + j   <--> i = n       // i -> depth
 *  n = i       <--> j = m - n   // j -> length
 *  p = k       <--> k = p       // k -> batch
 *
 *  Loop bounds:
 *  2 <= m < L + D - 1
 *  max(1, m - L + 1) <= n < min(m, D)
 *  0<= p < N
 */

void Region3DepthFirst(const GPUContext& context, Tensor& hsss, Tensor& csss,
                       const Tensor& xss, const Tensor& ws, const Tensor& us,
                       std::shared_ptr<Allocator> allocator,
                       std::vector<float>& time_info, bool fuse_bmm = false) {
    const int64_t kDepth1 = ws.dim_size(0);  // the 1st control attached to wss
    if (kDepth1 == 1) return;

    const int64_t kDepth2 = xss.dim_size(0);  // the 2rd control attached to xss
    const int64_t kDepth3 = xss.dim_size(1);  // the 3rd control attached to xss

    // size of tensor (vectors in this case) stored in FracatlTensors.
    const int64_t kVecSize = hsss.dim_size(-1);

    int64_t lb3 = 2;
    int64_t ub3 = xss.dim_size(0) + ws.dim_size(0) - 1;

    int64_t lb1 = 0;
    int64_t ub1 = xss.dim_size(1) /*batch*/;

    auto cal_offset = [&](const std::string debug_info,
                          const std::vector<int64_t>& ids) {
        /* Given the index of the first point in a parallelizable
         * hyperplane, returns the offset (counted in bytes) of this
         * point. NOTE: the index MUST BE in the original addressing
         * space.
         */
        int64_t offset = (ids[0] * kDepth2 + ids[1]) * kDepth3 * kVecSize;
        return offset;
    };

    std::vector<float> compute(2, 0.);
    float total_scatter = 0.f;

    for (int64_t m = lb3; m < ub3; ++m) {  // the sequential loop
        // ============= Begin Prologue of a Control Region
        // ======================
        const int64_t lb2 = max(1l, m - kDepth2 + 1l);
        const int64_t ub2 = min(kDepth1, m);
        const int64_t length_depth2 =
            ub2 - lb2;  // the number of activated depths.

        /* Time stamp for the first point in the hyperplane is:
           m = m,                  // index of the hyperplane.
           n = max(1, m - L + 1),  // the lower bound of the second
           loop. p = 0                   // the lower bound of the thrid
           loop.

           i = n
           j = m - n
           k = p
         */
        int64_t j0 = max(1l, m - kDepth2 + 1);
        // `xs` refers to `hsss`[i - 1][j][k]
        std::vector<int64_t> start_xs{j0 - 1, m - j0, 0};
        // `hs` refers to `hsss`[i][j - 1][k]
        std::vector<int64_t> start_hs{j0, m - j0 - 1, 0};

        int64_t stride = kDepth2 - 1;
        int64_t strided_span = (length_depth2 - 1) * stride + 1;

        Tensor x({strided_span, hsss.dim_size(-2), hsss.dim_size(-1)}, nullptr);
        x.CreateFrom<float>(hsss, cal_offset("xs", start_xs));

        Tensor h({strided_span, hsss.dim_size(-2), hsss.dim_size(-1)}, nullptr);
        h.CreateFrom<float>(hsss, cal_offset("hs", start_hs));

        Tensor c({strided_span, csss.dim_size(-2), csss.dim_size(-1)}, nullptr);
        c.CreateFrom<float>(csss, cal_offset("cs", start_hs));

        Tensor w({length_depth2, ws.dim_size(1), ws.dim_size(2)}, nullptr);
        w.CreateFrom<float>(ws, lb2 * ws.dim_size(1) * ws.dim_size(2));

        Tensor u({length_depth2, us.dim_size(1), us.dim_size(2)}, nullptr);
        u.CreateFrom<float>(us, lb2 * us.dim_size(1) * us.dim_size(2));

        // tmp1 = x @ W[i,f,o,c]
        Tensor tmp1({length_depth2, kDepth3, ws.dim_size(-1)}, allocator);
        // tmp2 = h @ U[i,f,o,c]
        Tensor tmp2({length_depth2, kDepth3, ws.dim_size(-1)}, allocator);
        FillZeros(tmp1);
        FillZeros(tmp2);

        Tensor tmp3({length_depth2, kDepth3, ws.dim_size(-1)}, allocator);
        Tensor tmp4({length_depth2, kDepth3, ws.dim_size(-1)}, allocator);
        FillZeros(tmp3);
        FillZeros(tmp4);

        Tensor out1({length_depth2, kDepth3, csss.dim_size(-1)}, allocator);
        Tensor out2({length_depth2, kDepth3, hsss.dim_size(-1)}, allocator);
        FillZeros(out1);
        FillZeros(out2);

        int end_size = 3;  // index over the first 3 dimensionalities, the
                           // remaining dimensions are to slice over.
        int numel = length_depth2 * kDepth3;
        int64_t* scatter_indices_cpu =
            (int64_t*)malloc(numel * end_size * sizeof(int64_t));

        int count = 0;
        for (int64_t n = lb2; n < ub2; ++n) {
            for (int64_t p = lb1; p < ub1; ++p) {
                scatter_indices_cpu[count * end_size] = n;
                scatter_indices_cpu[count * end_size + 1] = m - n;
                scatter_indices_cpu[count * end_size + 2] = p;
                count += 1;
            }
        }

        int64_t* scatter_indices;
        CudaCheck(
            cudaMalloc(&scatter_indices, numel * end_size * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(scatter_indices, scatter_indices_cpu,
                             numel * end_size * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

        // ============= End Prologue of a Control Region
        // ======================
        int m1 = xss.dim_size(1) /*batch*/;
        int n1 = w.dim_size(-1) /*hidden*/;
        int k1 = xss.dim_size(2) /*hidden*/;

        stride *= (hsss.dim_size(-1) * hsss.dim_size(-2));
        std::vector<float> times;
        if (fuse_bmm) {
            // batch_count = length_depth2
            // stride = kDepth2 - 1
            // m1: batch_size
            // n1: hidden_size
            // k1: hidden_size
            times =
                LstmCellFuseBMM(context, x, w, c, h, u, out1, out2, tmp1, tmp2,
                                tmp3, tmp4, m1, n1, k1, stride, length_depth2);
        } else {
            times = LstmCellBMM(context, x, w, c, h, u, out1, out2, tmp1, tmp2,
                                tmp3, tmp4, m1, n1, k1, stride, stride,
                                length_depth2);
        }

        compute[0] += times[0];
        compute[1] += times[1];

        // Begin Epilogue of a Control Region, Reversed Access
        int64_t* input_dims_ysss;
        CudaCheck(cudaMalloc(&input_dims_ysss, hsss.ndim() * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(input_dims_ysss, hsss.dims().data(),
                             hsss.ndim() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

        int slice_size = hsss.dim_size(-1);
        int block = 512;
        int num = slice_size * numel;
        int grid = (num + block - 1) / block;

        float elapsed_scatter = 0.f;

        cudaEvent_t start_scatter, stop_scatter;
        cudaEventCreate(&start_scatter);
        cudaEventCreate(&stop_scatter);

        cudaEventRecord(start_scatter, 0);
        cuda_kernel::ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            out1.data<float>(), scatter_indices, csss.mutable_data<float>(),
            input_dims_ysss, numel, slice_size, end_size);
        cuda_kernel::ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            out2.data<float>(), scatter_indices, hsss.mutable_data<float>(),
            input_dims_ysss, numel, slice_size, end_size);
        GpuElapse(start_scatter, stop_scatter, elapsed_scatter, total_scatter);

        free(scatter_indices_cpu);
        cudaFree(scatter_indices);
        cudaFree(input_dims_ysss);
    }

    time_info[0] += 0.;             // gather
    time_info[1] += compute[0];     // gemm
    time_info[2] += compute[1];     // elementwise
    time_info[3] += total_scatter;  // scatter
}

}  // namespace core
}  // namespace kaleido
