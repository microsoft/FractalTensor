#pragma once

#include "utils.h"

namespace kaleido {
namespace core {

void Region4DepthFirst(const GPUContext& context, Tensor& hsss,
                       const Tensor& xss, const Tensor& yss, const Tensor& ws,
                       const Tensor& us, const Tensor& bs,
                       std::shared_ptr<Allocator> allocator,
                       std::vector<float>& time_info) {
    if (xss.dim_size(0) <= 1) return;
    if (yss.dim_size(0) <= 1) return;

    ops::PrintOp<GPUContext, CUDAPlace, float> printer;

    std::vector<float> compute(2, 0.);
    float total_gather = 0.f;
    float total_scatter = 0.f;

    /* ============ Region 4 is controlled by s single control
     ================ (0, >0, >0, *)
     [[1, 0, 0, 0]
      [0, 1, 1, 0]
      [0, 1, 0, 0]
      [0, 0, 0, 1]]
      m = 0           <--> d = 0          // d -> depth
      n = i + j       <--> i = p          // i -> src_length
      p = i           <--> j = n - p      // j -> trg_length
      q = k           <--> k = q          // k -> batch

      Loop bounds:
      2 <= n < SL + TL - 1
      max(1, n - TL + 1) <= p < min(n, SL)
      0 <= q < N

      Loop invariant inputs:
      w, u, b

      Gather inputs:
      state_x, state_y

      Fixed strides:
      x_t, y_t

      Scatter outputs:
      h_x, h_y
      ==========================================================================
   */

    int64_t depth_size = ws.dim_size(0);
    int64_t src_length = hsss.dim_size(1);
    int64_t trg_length = hsss.dim_size(2);
    int64_t batch_size = xss.dim_size(-2);
    int64_t hidden_size = ws.dim_size(-1);

    int kGridDim = 2;
    Tensor w({1, hidden_size, hidden_size}, nullptr);
    Tensor u({1, kGridDim * hidden_size, hidden_size}, nullptr);
    Tensor b({1, batch_size, hidden_size}, nullptr);
    w.CreateFrom<float>(ws, 0);
    u.CreateFrom<float>(us, 0);
    b.CreateFrom<float>(bs, 0);

    // Loop bounds: 2 <= n < SL + TL - 1
    int64_t lb0 = 2;
    int64_t ub0 = src_length + trg_length - 1;

    // Loop bounds: 0 <= q < N
    int64_t lb2 = 0;
    int64_t ub2 = batch_size;

    for (int64_t n = lb0; n < ub0; ++n) {
        // Loop bounds: max(1, n - TL + 1) <= p < min(n, SL)
        int64_t lb1 = max(1l, n - trg_length + 1l);
        int64_t ub1 = min(src_length, n);
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
        for (int64_t p = lb1; p < ub1; ++p) {
            for (int64_t q = lb2; q < ub2; ++q) {
                // state_x,state_y
                gather_indices_h_cpu[0][count * end_size_h] = 0;
                gather_indices_h_cpu[0][count * end_size_h + 1] = p - 1;
                gather_indices_h_cpu[0][count * end_size_h + 2] = n - p;
                gather_indices_h_cpu[0][count * end_size_h + 3] = 0;
                gather_indices_h_cpu[0][count * end_size_h + 4] = q;

                gather_indices_h_cpu[1][count * end_size_h] = 0;
                gather_indices_h_cpu[1][count * end_size_h + 1] = p;
                gather_indices_h_cpu[1][count * end_size_h + 2] = n - p - 1;
                gather_indices_h_cpu[1][count * end_size_h + 3] = 1;
                gather_indices_h_cpu[1][count * end_size_h + 4] = q;

                // h_x,h_y
                gather_indices_h_cpu[2][count * end_size_h] = 0;
                gather_indices_h_cpu[2][count * end_size_h + 1] = p;
                gather_indices_h_cpu[2][count * end_size_h + 2] = n - p;
                gather_indices_h_cpu[2][count * end_size_h + 3] = 0;
                gather_indices_h_cpu[2][count * end_size_h + 4] = q;

                gather_indices_h_cpu[3][count * end_size_h] = 0;
                gather_indices_h_cpu[3][count * end_size_h + 1] = p;
                gather_indices_h_cpu[3][count * end_size_h + 2] = n - p;
                gather_indices_h_cpu[3][count * end_size_h + 3] = 1;
                gather_indices_h_cpu[3][count * end_size_h + 4] = q;

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

        Tensor x_t({numel_dim1, batch_size, hidden_size}, nullptr);
        Tensor y_t({numel_dim1, batch_size, hidden_size}, nullptr);
        x_t.CreateFrom<float>(xss, lb1 * batch_size * hidden_size);
        y_t.CreateFrom<float>(yss, (n - ub1) * batch_size * hidden_size);

        Tensor state_x({numel_dim1, batch_size, hidden_size}, allocator);
        Tensor state_y({numel_dim1, batch_size, hidden_size}, allocator);
        FillZeros(state_x);
        FillZeros(state_y);

        Tensor state({numel_dim1, batch_size, 2 * hidden_size}, allocator);
        FillZeros(state);

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
            state_y.mutable_data<float>(), numel, hsss.dim_size(-1),
            end_size_h);

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
        auto times = VanillaRNNCellBMM(
            context, x_t, y_t, state, w, u, b, tmp1, tmp2, tmp3, tmp4, h_x, h_y,
            numel_dim1 * batch_size, hidden_size, hidden_size, 1);

        compute[0] += times[0];
        compute[1] += times[1];
        CHECK_ERROR(cudaGetLastError());

        float elapsed_scatter = 0.f;
        cudaEvent_t start_scatter, stop_scatter;
        cudaEventCreate(&start_scatter);
        cudaEventCreate(&stop_scatter);
        cudaEventRecord(start_scatter, 0);
        cuda_kernel::ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            h_x.data<float>(), gather_indices_h[2], hsss.mutable_data<float>(),
            input_dims_hsss, numel, slice_size, end_size_h);
        cuda_kernel::ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            h_y.data<float>(), gather_indices_h[3], hsss.mutable_data<float>(),
            input_dims_hsss, numel, slice_size, end_size_h);
        GpuElapse(start_scatter, stop_scatter, elapsed_scatter, total_scatter);
        CHECK_ERROR(cudaGetLastError());

        for (int i = 0; i < num_indices_h; ++i) {
            free(gather_indices_h_cpu[i]);
            cudaFree(gather_indices_h[i]);
        }
    }

#ifdef DEBUG_ON
    std::cout << "region4: hsss:" << std::endl;
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
