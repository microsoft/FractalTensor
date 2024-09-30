#pragma once

#include "utils.h"

namespace kaleido {
namespace core {

void Region8DepthFirst(const GPUContext& context, Tensor& hsss,
                       const Tensor& xss, const Tensor& yss, const Tensor& ws,
                       const Tensor& us, const Tensor& bs,
                       std::shared_ptr<Allocator> allocator,
                       std::vector<float>& time_info) {
    if (ws.dim_size(0) <= 1) return;
    if (xss.dim_size(0) <= 1) return;
    if (yss.dim_size(0) <= 1) return;

    ops::PrintOp<GPUContext, CUDAPlace, float> printer;

    std::vector<float> compute(2, 0.);
    float total_gather = 0.f;
    float total_scatter = 0.f;

    /* ============ Region 8 is controlled by s single control
     ================
     (>0, >0, >0, *)
     [[1, 1, 1, 0]
      [1, 0, 0, 0]
      [0, 1, 0, 0]
      [0, 0, 0, 1]]
      m = d + i + j   <--> d = n          // d -> depth
      n = d           <--> i = p          // i -> src_length
      p = i           <--> j = m - n - p  // j -> trg_length
      q = k           <--> k = q          //k -> batch

      Loop bounds:
      3 <= m < D + SL + TL - 2
      max(1, m - SL - TL + 2) <= n < min(m - 1, D)
      max(1, m - n - TL + 1) <= p < min(m - n, SL)
      0 <= q < N

      Loop invariant inputs:

      Gather inputs:
      x_t, y_t, state_x, state_y, w, u, b

      Fixed strides:

      Scatter outputs:
      h_x, h_y
      ========================================================================*/

    int64_t depth_size = ws.dim_size(0);
    int64_t src_length = hsss.dim_size(1);
    int64_t trg_length = hsss.dim_size(2);
    int64_t batch_size = xss.dim_size(-2);
    int64_t hidden_size = ws.dim_size(-1);

    // Loop bounds: 3 <= m < D + SL + TL - 2
    int64_t lb0 = 3;
    int64_t ub0 = depth_size + src_length + trg_length - 2;

    // Loop bounds: 0 <= q < N
    int64_t lb3 = 0;
    int64_t ub3 = batch_size;

    for (int64_t m = lb0; m < ub0; ++m) {
        // Loop bounds: max(1, m - SL - TL + 2) <= n < min(m - 1, D)
        int64_t lb1 = max(1l, m - src_length - trg_length + 2l);
        int64_t ub1 = min(m - 1, depth_size);
        int64_t numel_dim1 = ub1 - lb1;

        int numel = 0;
        int end_size_h = 5;
        int end_size_w = 1;
        int num_indices_h = 6;
        int num_indices_w = 1;

        std::vector<int64_t*> gather_indices_h_cpu(num_indices_h, nullptr);
        for (int i = 0; i < num_indices_h; ++i) {
            gather_indices_h_cpu[i] =
                (int64_t*)malloc(numel_dim1 * min(m, src_length) * batch_size *
                                 end_size_h * sizeof(int64_t));
        }

        std::vector<int64_t*> gather_indices_w_cpu(1, nullptr);
        for (int i = 0; i < 1; ++i) {
            gather_indices_w_cpu[i] =
                (int64_t*)malloc(numel_dim1 * min(m, src_length) * batch_size *
                                 end_size_h * sizeof(int64_t));
        }

        int64_t* gather_indices_b_cpu =
            (int64_t*)malloc(numel_dim1 * min(m, src_length) * batch_size *
                             end_size_h * sizeof(int64_t));

        for (int64_t n = lb1; n < ub1; ++n) {
            // Loop bounds: max(1, m - n - TL + 1) <= p < min(m - n, SL)
            int64_t lb2 = max(1l, m - n - trg_length + 1l);
            int64_t ub2 = min(m - n, src_length);
            int64_t numel_dim2 = ub2 - lb2;

            int count = numel;
            for (int64_t p = lb2; p < ub2; ++p) {
                // w u b
                gather_indices_w_cpu[0][count / batch_size * end_size_w] = n;

                for (int64_t q = lb3; q < ub3; ++q) {
                    // x_t,y_t
                    gather_indices_h_cpu[0][count * end_size_h] = n - 1;
                    gather_indices_h_cpu[0][count * end_size_h + 1] = p;
                    gather_indices_h_cpu[0][count * end_size_h + 2] = m - n - p;
                    gather_indices_h_cpu[0][count * end_size_h + 3] = 0;
                    gather_indices_h_cpu[0][count * end_size_h + 4] = q;

                    gather_indices_h_cpu[1][count * end_size_h] = n - 1;
                    gather_indices_h_cpu[1][count * end_size_h + 1] = p;
                    gather_indices_h_cpu[1][count * end_size_h + 2] = m - n - p;
                    gather_indices_h_cpu[1][count * end_size_h + 3] = 1;
                    gather_indices_h_cpu[1][count * end_size_h + 4] = q;

                    // state_x,state_y
                    gather_indices_h_cpu[2][count * end_size_h] = n;
                    gather_indices_h_cpu[2][count * end_size_h + 1] = p - 1;
                    gather_indices_h_cpu[2][count * end_size_h + 2] = m - n - p;
                    gather_indices_h_cpu[2][count * end_size_h + 3] = 0;
                    gather_indices_h_cpu[2][count * end_size_h + 4] = q;

                    gather_indices_h_cpu[3][count * end_size_h] = n;
                    gather_indices_h_cpu[3][count * end_size_h + 1] = p;
                    gather_indices_h_cpu[3][count * end_size_h + 2] =
                        m - n - p - 1;
                    gather_indices_h_cpu[3][count * end_size_h + 3] = 1;
                    gather_indices_h_cpu[3][count * end_size_h + 4] = q;

                    // h_x,h_y
                    gather_indices_h_cpu[4][count * end_size_h] = n;
                    gather_indices_h_cpu[4][count * end_size_h + 1] = p;
                    gather_indices_h_cpu[4][count * end_size_h + 2] = m - n - p;
                    gather_indices_h_cpu[4][count * end_size_h + 3] = 0;
                    gather_indices_h_cpu[4][count * end_size_h + 4] = q;

                    gather_indices_h_cpu[5][count * end_size_h] = n;
                    gather_indices_h_cpu[5][count * end_size_h + 1] = p;
                    gather_indices_h_cpu[5][count * end_size_h + 2] = m - n - p;
                    gather_indices_h_cpu[5][count * end_size_h + 3] = 1;
                    gather_indices_h_cpu[5][count * end_size_h + 4] = q;

                    gather_indices_b_cpu[count * end_size_w] = n;
                    count += 1;
                }
            }
            numel += numel_dim2 * batch_size;
        }

        std::vector<int64_t*> gather_indices_h(num_indices_h, nullptr);
        std::vector<int64_t*> gather_indices_w(num_indices_w, nullptr);
        int64_t* gather_indices_b;

        for (int i = 0; i < num_indices_h; ++i) {
            CudaCheck(cudaMalloc(&gather_indices_h[i],
                                 numel * end_size_h * sizeof(int64_t)));
            CudaCheck(cudaMemcpy(gather_indices_h[i], gather_indices_h_cpu[i],
                                 numel * end_size_h * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));
        }
        for (int i = 0; i < num_indices_w; ++i) {
            CudaCheck(cudaMalloc(&gather_indices_w[i],
                                 numel * end_size_w * sizeof(int64_t)));
            CudaCheck(cudaMemcpy(gather_indices_w[i], gather_indices_w_cpu[i],
                                 numel * end_size_w * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));
        }

        CudaCheck(cudaMalloc(&gather_indices_b,
                             numel * end_size_w * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(gather_indices_b, gather_indices_b_cpu,
                             numel * end_size_w * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

        int64_t* input_dims_hsss;
        CudaCheck(cudaMalloc(&input_dims_hsss, hsss.ndim() * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(input_dims_hsss, hsss.dims().data(),
                             hsss.ndim() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

        int64_t* input_dims_ws;
        CudaCheck(cudaMalloc(&input_dims_ws, ws.ndim() * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(input_dims_ws, ws.dims().data(),
                             ws.ndim() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));
        int64_t* input_dims_us;
        CudaCheck(cudaMalloc(&input_dims_us, us.ndim() * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(input_dims_us, us.dims().data(),
                             us.ndim() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));
        int64_t* input_dims_bs;
        CudaCheck(cudaMalloc(&input_dims_bs, bs.ndim() * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(input_dims_bs, bs.dims().data(),
                             bs.ndim() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

        Tensor x_t({numel / batch_size, batch_size, hidden_size}, allocator);
        Tensor y_t({numel / batch_size, batch_size, hidden_size}, allocator);
        FillZeros(x_t);
        FillZeros(y_t);

        Tensor state_x({numel / batch_size, batch_size, hidden_size},
                       allocator);
        Tensor state_y({numel / batch_size, batch_size, hidden_size},
                       allocator);
        FillZeros(state_x);
        FillZeros(state_y);

        Tensor state({numel / batch_size, batch_size, 2 * hidden_size},
                     allocator);
        FillZeros(state);

        int kGridDim = 2;
        Tensor w({numel / batch_size, hidden_size, hidden_size}, allocator);
        Tensor u({numel / batch_size, kGridDim * hidden_size, hidden_size},
                 allocator);
        Tensor b({numel / batch_size, batch_size, hidden_size}, allocator);
        FillZeros(w);
        FillZeros(u);
        FillZeros(b);

        Tensor tmp1({numel / batch_size, batch_size, hidden_size}, allocator);
        Tensor tmp2({numel / batch_size, batch_size, hidden_size}, allocator);
        Tensor tmp3({numel / batch_size, batch_size, hidden_size}, allocator);
        Tensor tmp4({numel / batch_size, batch_size, hidden_size}, allocator);
        FillZeros(tmp1);
        FillZeros(tmp2);
        FillZeros(tmp3);
        FillZeros(tmp4);

        Tensor h_x({numel / batch_size, batch_size, hidden_size}, allocator);
        Tensor h_y({numel / batch_size, batch_size, hidden_size}, allocator);
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
            x_t.mutable_data<float>(), numel, hsss.dim_size(-1), end_size_h);
        cuda_kernel::GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            hsss.data<float>(), input_dims_hsss, gather_indices_h[1],
            y_t.mutable_data<float>(), numel, hsss.dim_size(-1), end_size_h);
        cuda_kernel::GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            hsss.data<float>(), input_dims_hsss, gather_indices_h[2],
            state_x.mutable_data<float>(), numel, hsss.dim_size(-1),
            end_size_h);
        cuda_kernel::GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            hsss.data<float>(), input_dims_hsss, gather_indices_h[3],
            state_y.mutable_data<float>(), numel, hsss.dim_size(-1),
            end_size_h);

        cuda_kernel::GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            ws.data<float>(), input_dims_ws, gather_indices_w[0],
            w.mutable_data<float>(), numel / batch_size,
            ws.dim_size(-1) * ws.dim_size(-2), end_size_w);
        cuda_kernel::GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            us.data<float>(), input_dims_us, gather_indices_w[0],
            u.mutable_data<float>(), numel / batch_size,
            us.dim_size(-1) * us.dim_size(-2), end_size_w);
        cuda_kernel::GatherNdCUDAKernel<float><<<grid, block, 0>>>(
            bs.data<float>(), input_dims_bs, gather_indices_b,
            b.mutable_data<float>(), numel, bs.dim_size(-1), end_size_w);
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
            batch_size, hidden_size, hidden_size, numel / batch_size);

        compute[0] += times[0];
        compute[1] += times[1];
        CHECK_ERROR(cudaGetLastError());

        float elapsed_scatter = 0.f;
        cudaEvent_t start_scatter, stop_scatter;
        cudaEventCreate(&start_scatter);
        cudaEventCreate(&stop_scatter);
        cudaEventRecord(start_scatter, 0);
        cuda_kernel::ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            h_x.data<float>(), gather_indices_h[4], hsss.mutable_data<float>(),
            input_dims_hsss, numel, slice_size, end_size_h);
        cuda_kernel::ScatterNdCUDAKernel<float><<<grid, block, 0>>>(
            h_y.data<float>(), gather_indices_h[5], hsss.mutable_data<float>(),
            input_dims_hsss, numel, slice_size, end_size_h);
        GpuElapse(start_scatter, stop_scatter, elapsed_scatter, total_scatter);
        CHECK_ERROR(cudaGetLastError());

        for (int i = 0; i < num_indices_h; ++i) {
            free(gather_indices_h_cpu[i]);
            cudaFree(gather_indices_h[i]);
        }
        for (int i = 0; i < num_indices_w; ++i) {
            free(gather_indices_w_cpu[i]);
            cudaFree(gather_indices_w[i]);
        }
        free(gather_indices_b_cpu);
        cudaFree(gather_indices_b);
    }

#ifdef DEBUG_ON
    std::cout << "region8: hsss:" << std::endl;
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
