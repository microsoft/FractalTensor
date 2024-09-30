#pragma once

#include "utils.h"

namespace kaleido {
namespace core {

void Region6DepthFirst(const GPUContext& context, Tensor& hsss,
                       const Tensor& xss, const Tensor& yss, const Tensor& ws,
                       const Tensor& us, const Tensor& bs,
                       std::shared_ptr<Allocator> allocator,
                       std::vector<float>& time_info) {
    if (ws.dim_size(0) <= 1) return;
    if (xss.dim_size(0) <= 1) return;

    ops::PrintOp<GPUContext, CUDAPlace, float> printer;

    std::vector<float> compute(2, 0.);
    float total_gather = 0.f;
    float total_scatter = 0.f;

    /* ============ Region 6 is controlled by s single control
     ================
     (>0, >0, 0, *)
     [[1, 1, 0, 0]
      [1, 0, 0, 0]
      [0, 0, 1, 0]
      [0, 0, 0, 1]]
      m = d + i       <--> d = n          // d -> depth
      n = d           <--> i = m - n      // i -> src_length
      p = 0           <--> j = 0          // j -> trg_length
      q = k           <--> k = q          // k -> batch

      Loop bounds:
      2 <= m < D + SL - 1
      max(1, m - SL + 1) <= n < min(m, D)
      0 <= q < N

      Loop invariant inputs:

      Gather inputs:
      x_t, y_t, state_x

      Fixed strides:
      w, u, b

      Scatter outputs:
      h_x, h_y
      ==========================================================================
   */

    int64_t depth_size = ws.dim_size(0);
    int64_t src_length = hsss.dim_size(1);
    int64_t trg_length = hsss.dim_size(2);
    int64_t batch_size = xss.dim_size(-2);
    int64_t hidden_size = ws.dim_size(-1);

    // Loop bounds: 2 <= m < D + SL - 1
    int64_t lb0 = 2;
    int64_t ub0 = depth_size + src_length - 1;

    // Loop bounds: 0 <= q < N
    int64_t lb2 = 0;
    int64_t ub2 = batch_size;

    for (int64_t m = lb0; m < ub0; ++m) {
        // Loop bounds: max(1, m - SL + 1) <= n < min(m, D)
        int64_t lb1 = max(1l, m - src_length + 1l);
        int64_t ub1 = min(depth_size, m);
        int64_t numel_dim1 = ub1 - lb1;

        int num_indices_h = 6;
        int end_size_h = 5;
        int numel = numel_dim1 * (ub2 - lb2);
        std::vector<int64_t*> gather_indices_h_cpu(num_indices_h, nullptr);

        for (int i = 0; i < num_indices_h; ++i) {
            gather_indices_h_cpu[i] =
                (int64_t*)malloc(numel * end_size_h * sizeof(int64_t));
        }

        int count = 0;
        for (int64_t n = lb1; n < ub1; ++n) {
            for (int64_t q = lb2; q < ub2; ++q) {
                // state_x
                gather_indices_h_cpu[0][count * end_size_h] = n;
                gather_indices_h_cpu[0][count * end_size_h + 1] = m - n - 1;
                gather_indices_h_cpu[0][count * end_size_h + 2] = 0;
                gather_indices_h_cpu[0][count * end_size_h + 3] = 0;
                gather_indices_h_cpu[0][count * end_size_h + 4] = q;

                // x_t,y_t
                gather_indices_h_cpu[1][count * end_size_h] = n - 1;
                gather_indices_h_cpu[1][count * end_size_h + 1] = m - n;
                gather_indices_h_cpu[1][count * end_size_h + 2] = 0;
                gather_indices_h_cpu[1][count * end_size_h + 3] = 0;
                gather_indices_h_cpu[1][count * end_size_h + 4] = q;

                gather_indices_h_cpu[2][count * end_size_h] = n - 1;
                gather_indices_h_cpu[2][count * end_size_h + 1] = m - n;
                gather_indices_h_cpu[2][count * end_size_h + 2] = 0;
                gather_indices_h_cpu[2][count * end_size_h + 3] = 1;
                gather_indices_h_cpu[2][count * end_size_h + 4] = q;

                // h_x,h_y
                gather_indices_h_cpu[3][count * end_size_h] = n;
                gather_indices_h_cpu[3][count * end_size_h + 1] = m - n;
                gather_indices_h_cpu[3][count * end_size_h + 2] = 0;
                gather_indices_h_cpu[3][count * end_size_h + 3] = 0;
                gather_indices_h_cpu[3][count * end_size_h + 4] = q;

                gather_indices_h_cpu[4][count * end_size_h] = n;
                gather_indices_h_cpu[4][count * end_size_h + 1] = m - n;
                gather_indices_h_cpu[4][count * end_size_h + 2] = 0;
                gather_indices_h_cpu[4][count * end_size_h + 3] = 1;
                gather_indices_h_cpu[4][count * end_size_h + 4] = q;

                count += 1;
            }
        }

        std::vector<int64_t*> gather_indices_h(num_indices_h, nullptr);
        for (int i = 0; i < num_indices_h; ++i) {
            CudaCheck(cudaMalloc(&gather_indices_h[i],
                                 numel * end_size_h * sizeof(int64_t)));
            CudaCheck(cudaMemcpy(gather_indices_h[i], gather_indices_h_cpu[i],
                                 numel * end_size_h * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));
        }

        int64_t* input_dims_hsss;
        CudaCheck(cudaMalloc(&input_dims_hsss, hsss.ndim() * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(input_dims_hsss, hsss.dims().data(),
                             hsss.ndim() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

        Tensor x_t({numel_dim1, batch_size, hidden_size}, allocator);
        Tensor y_t({numel_dim1, batch_size, hidden_size}, allocator);
        FillZeros(x_t);
        FillZeros(y_t);

        Tensor state_x({numel_dim1, batch_size, hidden_size}, allocator);
        Tensor state_y({numel_dim1, batch_size, hidden_size}, allocator);
        FillZeros(state_x);
        FillZeros(state_y);

        Tensor state({numel_dim1, batch_size, 2 * hidden_size}, allocator);
        FillZeros(state);

        int kGridDim = 2;
        Tensor w({numel_dim1, hidden_size, hidden_size}, nullptr);
        Tensor u({numel_dim1, kGridDim * hidden_size, hidden_size}, nullptr);
        Tensor b({numel_dim1, batch_size, hidden_size}, nullptr);
        w.CreateFrom<float>(ws, (lb1)*hidden_size * hidden_size);
        u.CreateFrom<float>(us, (lb1)*kGridDim * hidden_size * hidden_size);
        b.CreateFrom<float>(bs, (lb1)*batch_size * hidden_size);

        Tensor tmp1({numel_dim1, batch_size, hidden_size}, allocator);
        Tensor tmp2({numel_dim1, batch_size, hidden_size}, allocator);
        Tensor tmp3({numel_dim1, batch_size, hidden_size}, allocator);
        Tensor tmp4({numel_dim1, batch_size, hidden_size}, allocator);
        FillZeros(tmp1);
        FillZeros(tmp2);
        FillZeros(tmp3);
        FillZeros(tmp4);

        Tensor h_x({numel_dim1, batch_size, hidden_size}, allocator);
        Tensor h_y({numel_dim1, batch_size, hidden_size}, allocator);
        FillZeros(h_x);
        FillZeros(h_y);

        int slice_size = hsss.dim_size(-1);
        int block = 512;
        int num = slice_size * numel;
        int grid = (num + block - 1) / block;

        float elapsed_gather = 0.f;
        cudaEvent_t start_gather, stop_gather;
        cudaEventCreate(&start_gather);
        cudaEventCreate(&stop_gather);
        cudaEventRecord(start_gather, 0);
        cuda_kernel::GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            hsss.data<float>(), input_dims_hsss, gather_indices_h[0],
            state_x.mutable_data<float>(), numel, hsss.dim_size(-1),
            end_size_h);
        cuda_kernel::GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            hsss.data<float>(), input_dims_hsss, gather_indices_h[1],
            x_t.mutable_data<float>(), numel, hsss.dim_size(-1), end_size_h);
        cuda_kernel::GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            hsss.data<float>(), input_dims_hsss, gather_indices_h[2],
            y_t.mutable_data<float>(), numel, hsss.dim_size(-1), end_size_h);
        GpuElapse(start_gather, stop_gather, elapsed_gather, total_gather);
        CHECK_ERROR(cudaGetLastError());

        std::vector<Tensor> tensors;
        tensors.emplace_back(state_x);
        tensors.emplace_back(state_y);

        ops::ConcatOp<GPUContext, CUDAPlace, float> cat;
        cat(context, tensors, state, 2);

#ifdef DEBUG_ON
        std::cout << "wave " << n << " numel: " << numel << std::endl;
#endif
        auto times = VanillaRNNCellBMM(context, x_t, y_t, state, w, u, b, tmp1,
                                       tmp2, tmp3, tmp4, h_x, h_y, batch_size,
                                       hidden_size, hidden_size, numel_dim1);

        compute[0] += times[0];
        compute[1] += times[1];
        CHECK_ERROR(cudaGetLastError());

        float elapsed_scatter = 0.f;
        cudaEvent_t start_scatter, stop_scatter;
        cudaEventCreate(&start_scatter);
        cudaEventCreate(&stop_scatter);
        cudaEventRecord(start_scatter, 0);

        cuda_kernel::ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            h_x.data<float>(), gather_indices_h[3], hsss.mutable_data<float>(),
            input_dims_hsss, numel, slice_size, end_size_h);
        cuda_kernel::ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            h_y.data<float>(), gather_indices_h[4], hsss.mutable_data<float>(),
            input_dims_hsss, numel, slice_size, end_size_h);
        GpuElapse(start_scatter, stop_scatter, elapsed_scatter, total_scatter);
        CHECK_ERROR(cudaGetLastError());

        for (int i = 0; i < num_indices_h; ++i) {
            free(gather_indices_h_cpu[i]);
            cudaFree(gather_indices_h[i]);
        }
    }

#ifdef DEBUG_ON
    std::cout << "region6: hsss:" << std::endl;
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
