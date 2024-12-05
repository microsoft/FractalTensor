#pragma once
#include "kaleido/core/device/cuda_info.h"
#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/kernels/gemm_utils.h"
#include "kaleido/core/device/kernels/lstm/inner.h"
#include "kaleido/core/device/kernels/lstm_kernel_traits.h"
#include "kaleido/core/device/kernels/lstm_kernels.h"
#include "kaleido/core/device/kernels/lstm_ref.h"
#include "kaleido/core/tile_shape.h"

namespace kaleido::core::cuda_kernel {

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteLSTMGate {
    void operator()(const Element* W, const Element* x_t, const Element* U,
                    const Element* h_t_1, Element* O) {
        // Whole GEMM shape
        static const int kM = dim_size<0, WholeShape>;
        static const int kN = dim_size<1, WholeShape>;
        static const int kK = dim_size<2, WholeShape>;

        // CTA GEMM shape
        static const int kTM = dim_size<0, CtaTileShape>;
        static const int kTN = dim_size<1, CtaTileShape>;
        static const int kTK = dim_size<2, CtaTileShape>;

        static_assert(kTM % dim_size<0, WarpArrangement> == 0,
                      "the M dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");
        static_assert(kTN % dim_size<1, WarpArrangement> == 0,
                      "the N dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");

        using KeTraits =
            LSTMLayerTraits<Element, InstructionShape, ValueMnk,
                            WarpArrangement, CtaTileShape, WholeShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        auto kernel = &KeCuteLSTMGate<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;

        const int block_m = CeilDiv<kM, kTM>;
        const int block_n = CeilDiv<kN, kTN>;

#ifdef DEBUG
        std::cout << "block_m: " << block_m << ", block_n: " << block_n
                  << std::endl;
#endif

        const int kThreads = KeTraits::kThreads;

        dim3 gridDim(block_m, block_n);
        dim3 blockDim(kThreads, 1, 1);

        kernel<<<gridDim, blockDim, smem_size>>>(W, U, x_t, h_t_1, O);
    }
};

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteBatchedLSTMGate {
    void operator()(const Element* w, const Element* x, const Element* u,
                    const Element* h, Element* O, int stride_a, int stride_b,
                    int stride_c, int m, int n, int k, int b) {
        // Whole GEMM shape
        static const int kM = dim_size<0, WholeShape>;
        static const int kN = dim_size<1, WholeShape>;
        static const int kK = dim_size<2, WholeShape>;
        static const int kB = dim_size<3, WholeShape>;

        // CTA GEMM shape
        static const int kTM = dim_size<0, CtaTileShape>;
        static const int kTN = dim_size<1, CtaTileShape>;
        static const int kTK = dim_size<2, CtaTileShape>;

        static_assert(kTM % dim_size<0, WarpArrangement> == 0,
                      "the M dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");
        static_assert(kTN % dim_size<1, WarpArrangement> == 0,
                      "the N dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");

        using KeTraits =
            BatchedLSTMGateTraits<Element, InstructionShape, ValueMnk,
                                  WarpArrangement, CtaTileShape, WholeShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        auto cute_batched_lstm_gate = &KeCuteBatchedLSTMGate<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(cute_batched_lstm_gate,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_size);
        }

        const int block_m = CeilDiv<kM, kTM>;
        const int block_n = CeilDiv<kN, kTN>;

#ifdef DEBUG
        std::cout << "block_m: " << block_m << ", block_n: " << block_n
                  << std::endl;
#endif

        const int kThreads = KeTraits::kThreads;

        dim3 gridDim(block_m, block_n, kB);
        dim3 blockDim(kThreads, 1, 1);

        cute_batched_lstm_gate<<<gridDim, blockDim, smem_size>>>(
            w, u, x, h, O, stride_a, stride_b, stride_c, m, n, k, b);
    }
};

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
struct CuteDynamicLstmGate {
    void operator()(const Element* w, const Element* x, const Element* u,
                    const Element* h, Element* t, const int m, const int n,
                    const int k) {
        // Whole GEMM shape
        static const int kM = m;
        static const int kN = n;
        static const int kK = k;

        // CTA GEMM shape
        static const int kTM = dim_size<0, CtaTileShape>;
        static const int kTN = dim_size<1, CtaTileShape>;
        static const int kTK = dim_size<2, CtaTileShape>;

        static_assert(kTM % dim_size<0, WarpArrangement> == 0,
                      "the M dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");
        static_assert(kTN % dim_size<1, WarpArrangement> == 0,
                      "the N dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");

        using KeTraits =
            DynamicLstmGateTraits<Element, InstructionShape, ValueMnk,
                                  WarpArrangement, CtaTileShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        auto cute_batched_lstm_gate = &KeCuteDynamicLstmGate<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(cute_batched_lstm_gate,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_size);
        }

        const int block_m = (kM + kTM - 1) / kTM;
        const int block_n = (kN + kTN - 1) / kTN;

#ifdef DEBUG
        std::cout << "block_m: " << block_m << ", block_n: " << block_n
                  << std::endl;
#endif

        const int kThreads = KeTraits::kThreads;

        dim3 gridDim(block_m, block_n, 1);
        dim3 blockDim(kThreads, 1, 1);

        cute_batched_lstm_gate<<<gridDim, blockDim, smem_size>>>(w, u, x, h, t,
                                                                 m, n, k);
    }
};

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
struct CuteDynamicBatchedLSTMGate {
    void operator()(const Element* w, const Element* x, const Element* u,
                    const Element* h, Element* O, int stride_a, int stride_b,
                    int stride_c, int m, int n, int k, int b) {
        // Whole GEMM shape
        static const int kM = m;
        static const int kN = n;
        static const int kK = k;
        static const int kB = b;

        // CTA GEMM shape
        static const int kTM = dim_size<0, CtaTileShape>;
        static const int kTN = dim_size<1, CtaTileShape>;
        static const int kTK = dim_size<2, CtaTileShape>;

        static_assert(kTM % dim_size<0, WarpArrangement> == 0,
                      "the M dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");
        static_assert(kTN % dim_size<1, WarpArrangement> == 0,
                      "the N dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");

        using KeTraits =
            DynamicBatchedLSTMGateTraits<Element, InstructionShape, ValueMnk,
                                         WarpArrangement, CtaTileShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        auto cute_batched_lstm_gate =
            &KeCuteDynamicBatchedLstmGate<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(cute_batched_lstm_gate,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_size);
        }

        const int block_m = (kM + kTM - 1) / kTM;
        const int block_n = (kN + kTN - 1) / kTN;

#ifdef DEBUG
        std::cout << "block_m: " << block_m << ", block_n: " << block_n
                  << std::endl;
#endif

        const int kThreads = KeTraits::kThreads;

        dim3 gridDim(block_m, block_n, kB);
        dim3 blockDim(kThreads, 1, 1);

        cute_batched_lstm_gate<<<gridDim, blockDim, smem_size>>>(
            w, u, x, h, O, stride_a, stride_b, stride_c, m, n, k, b);
    }
};

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteLSTMCell {
    float operator()(const Element* w, const Element* xs, const Element* u,
                     const Element* c_1, const Element* h_1, Element* c,
                     Element* h, Element* O) {
        // Whole GEMM shape
        // TODO: kM = 4 * M
        // kM: 4 * hidden_size
        static const int kM = dim_size<0, WholeShape>;
        // kN: batch_size / batch_size * seq_length
        static const int kN = dim_size<1, WholeShape>;
        // kK: hidden_size
        static const int kK = dim_size<2, WholeShape>;

        static const int M = kM / 4;
        static const int N = kN;
        static const int K = kK;

        // CTA GEMM shape
        static const int kTM = dim_size<0, CtaTileShape>;
        static const int kTN = dim_size<1, CtaTileShape>;
        static const int kTK = dim_size<2, CtaTileShape>;

        static_assert(kTM % dim_size<0, WarpArrangement> == 0,
                      "the M dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");
        static_assert(kTN % dim_size<1, WarpArrangement> == 0,
                      "the N dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");

        Element* t;
        CudaCheck(cudaMalloc(&t, kM * kN * sizeof(Element)));

        using KeTraits =
            LSTMLayerTraits<Element, InstructionShape, ValueMnk,
                            WarpArrangement, CtaTileShape, WholeShape>;

        auto lstm_gate = &KeCuteLSTMGate<Element, KeTraits>;
        static int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(lstm_gate,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_size);
        }

        const int kThreads = KeTraits::kThreads;
        const int block_m = CeilDiv<M, kTM>;
        const int block_n = CeilDiv<N, kTN>;

        dim3 lstm_gate_grid_dim(block_m, block_n);
        dim3 lstm_gate_block_dim(kThreads, 1, 1);

        Element* output = t;

        CudaTimer timer;
        timer.Start();

        lstm_gate<<<lstm_gate_grid_dim, lstm_gate_block_dim, smem_size>>>(
            w, xs, u, h_1, output);

        float lstm_gate_time = timer.Stop();

#ifdef DEBUG
        std::cout << "lstm_gate_time: " << lstm_gate_time << "ms" << std::endl;
#endif

        auto lstm_element_wise = &KeLSTMElementWise<Element>;

        const Element* i = output;
        const Element* f = output + M * N;
        const Element* o = output + 2 * M * N;
        const Element* c_bar = output + 3 * M * N;

        int kMaxThreads = GetGPUMaxThreadsPerMultiProcessor(0);
        int size = M * N;
        int block_size = (size + kMaxThreads - 1) / kMaxThreads;
        dim3 element_wise_grid_dim(block_size, 1, 1);
        dim3 element_wise_block_dim(kMaxThreads, 1, 1);

        timer.Start();

        lstm_element_wise<<<element_wise_grid_dim, element_wise_block_dim>>>(
            i, f, o, c_bar, c_1, c, h, kMaxThreads, size);

        float lstm_element_wise_time = timer.Stop();

#ifdef DEBUG
        std::cout << "lstm_element_wise_time: " << lstm_element_wise_time
                  << "ms" << std::endl;
#endif
        return lstm_gate_time + lstm_element_wise_time;
    }
};

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
struct CuteDynamicLstmCell {
    float operator()(const Element* w, const Element* x, const Element* u,
                     const Element* c, const Element* h, Element* c_out,
                     Element* h_out, int m, int n, int k) {
        static const int kM = m;
        static const int kN = n;
        static const int kK = k;

        static const int M = kM / 4;
        static const int N = kN;
        static const int K = kK;

        float elapsed_time = 0;
        CudaTimer timer;

        // Cuda malloc for output
        Element* t;
        CudaCheck(cudaMalloc(&t, m * n * sizeof(Element)));

        using DynamicLstmGate =
            CuteDynamicLstmGate<Element, InstructionShape, ValueMnk,
                                WarpArrangement, CtaTileShape>;
        DynamicLstmGate dynamic_lstm_gate;

        Element* w_tmp;
        CudaCheck(cudaMalloc(&w_tmp, m * k * sizeof(Element)));
        Element* x_tmp;
        CudaCheck(cudaMalloc(&x_tmp, n * k * sizeof(Element)));
        Element* u_tmp;
        CudaCheck(cudaMalloc(&u_tmp, m * k * sizeof(Element)));
        Element* h_tmp;
        CudaCheck(cudaMalloc(&h_tmp, n * k * sizeof(Element)));

        timer.Start();
        dynamic_lstm_gate(w_tmp, x_tmp, u_tmp, h_tmp, t, m, n, k);
        elapsed_time += timer.Stop();

        const Element* i = t;
        const Element* f = t + M * N;
        const Element* o = t + 2 * M * N;
        const Element* c_bar = t + 3 * M * N;

        auto lstm_element_wise = &KeLSTMElementWise<Element>;

        int kMaxThreads = GetGPUMaxThreadsPerMultiProcessor(0);
        int size = M * N;
        int block_size = (size + kMaxThreads - 1) / kMaxThreads;
        dim3 element_wise_grid_dim(block_size, 1, 1);
        dim3 element_wise_block_dim(kMaxThreads, 1, 1);

        timer.Start();
        lstm_element_wise<<<element_wise_grid_dim, element_wise_block_dim>>>(
            i, f, o, c_bar, c, c_out, h_out, kMaxThreads, size);
        elapsed_time += timer.Stop();

        CudaCheck(cudaFree(t));
        CudaCheck(cudaFree(w_tmp));
        CudaCheck(cudaFree(x_tmp));
        CudaCheck(cudaFree(u_tmp));
        CudaCheck(cudaFree(h_tmp));

        return elapsed_time;
    }
};

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape>
struct CuteFusedBMMLSTMCell {
    float operator()(const Element* w, const Element* x, const Element* u,
                     const Element* c, const Element* h, Element* c_out,
                     Element* h_out, int depth, int seq_length, int batch_size,
                     int hidden_size, int stride_a, int stride_b, int stride_c,
                     int m, int n, int k, int b) {
        static const int kM = m;
        static const int kN = n;
        static const int kK = k;
        static const int kB = b;

        static const int M = kM / 4;
        static const int N = kN;
        static const int K = kK;

        float elapsed_time = 0;

        CudaTimer timer;

        // Cuda malloc for output
        Element* t;
        cudaMalloc(&t, kB * kM * kN * sizeof(Element));

        using DynamicBatchedLSTMGate =
            CuteDynamicBatchedLSTMGate<Element, InstructionShape, ValueMnk,
                                       WarpArrangement, CtaTileShape>;

        DynamicBatchedLSTMGate dynamic_batched_lstm_gate;

        timer.Start();
        dynamic_batched_lstm_gate(w, x, u, h, t, stride_a, stride_b, stride_c,
                                  m, n, k, b);
        elapsed_time += timer.Stop();

        const Element* i = t;
        const Element* f = t + M * N;
        const Element* o = t + 2 * M * N;
        const Element* c_bar = t + 3 * M * N;

        int kMaxThreads = GetGPUMaxThreadsPerMultiProcessor(0);
        int size = M * N;
        int block_size = (size + kMaxThreads - 1) / kMaxThreads;
        dim3 element_wise_grid_dim(block_size, 1, 1);
        dim3 element_wise_block_dim(kMaxThreads, 1, 1);

        auto lstm_element_wise = &KeLSTMElementWise<Element>;

        timer.Start();
        lstm_element_wise<<<element_wise_grid_dim, element_wise_block_dim>>>(
            i, f, o, c_bar, c, c_out, h_out, kMaxThreads, size);
        elapsed_time += timer.Stop();

        CudaCheck(cudaFree(t));

        return elapsed_time;
    }
};

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteLSTMLayer {
    float operator()(const Element* w, const Element* xs, const Element* u,
                     const Element* c_init, const Element* h_init,
                     Element* csss, Element* hsss, int seq_length,
                     float* time_ptr = nullptr) {
        // Whole GEMM shape
        static const int kM = dim_size<0, WholeShape>;
        static const int kN = dim_size<1, WholeShape>;
        static const int kK = dim_size<2, WholeShape>;

        static const int M = kM / 4;
        static const int N = kN;
        static const int K = kK;

        // CTA GEMM shape
        static const int kTM = dim_size<0, CtaTileShape>;
        static const int kTN = dim_size<1, CtaTileShape>;
        static const int kTK = dim_size<2, CtaTileShape>;

        static_assert(kTM % dim_size<0, WarpArrangement> == 0,
                      "the M dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");
        static_assert(kTN % dim_size<1, WarpArrangement> == 0,
                      "the N dimension of the CTA tile should be "
                      "divisible by the "
                      "number of warps along that that dimension.");

        int kMaxThreads = GetGPUMaxThreadsPerMultiProcessor(0);

        CudaTimer timer;

        Element* t;
        CudaCheck(cudaMalloc(&t, seq_length * kM * kN * sizeof(Element)));

        using KeTraits =
            LSTMLayerTraits<Element, InstructionShape, ValueMnk,
                            WarpArrangement, CtaTileShape, WholeShape>;

        static int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        auto lstm_gate = &KeCuteLSTMGate<Element, KeTraits>;
        auto lstm_element_wise = &KeLSTMElementWise<Element>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(lstm_gate,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem_size);
        }

        int size = M * N;
        int block_size = (size + kMaxThreads - 1) / kMaxThreads;

        const int block_m = CeilDiv<M, kTM>;
        const int block_n = CeilDiv<N, kTN>;
        const int kThreads = KeTraits::kThreads;

#ifdef DEBUG
        std::cout << "block_m: " << block_m << ", block_n: " << block_n
                  << "; threads: " << kThreads << std::endl;
#endif

        dim3 lstm_gate_grid_dim(block_m, block_n);
        dim3 lstm_gate_block_dim(kThreads, 1, 1);

        Element* output = t;

        timer.Start();

        lstm_gate<<<lstm_gate_grid_dim, lstm_gate_block_dim, smem_size>>>(
            w, xs, u, h_init, output);

        float time1 = timer.Stop();

        cudaDeviceSynchronize();

        const Element* i = output;
        const Element* f = output + M * N;
        const Element* o = output + 2 * M * N;
        const Element* c_bar = output + 3 * M * N;

        dim3 element_wise_grid_dim(block_size, 1, 1);
        dim3 element_wise_block_dim(kMaxThreads, 1, 1);

        timer.Start();

        lstm_element_wise<<<element_wise_grid_dim, element_wise_block_dim,
                            smem_size>>>(i, f, o, c_bar, c_init, csss, hsss,
                                         kMaxThreads, size);
        float time2 = timer.Stop();

        timer.Start();
        for (int i = 1; i < seq_length; ++i) {
            const Element* c_1 = csss + (i - 1) * M * N;
            const Element* h_1 = hsss + (i - 1) * M * N;
            const Element* x = xs + i * K * N;
            Element* t_offset = t + i * K * N;
            Element* c = csss + i * M * N;
            Element* h = hsss + i * M * N;

            lstm_gate<<<lstm_gate_grid_dim, lstm_gate_block_dim, smem_size>>>(
                w, u, x, c_1, t_offset);

            cudaDeviceSynchronize();

            const Element* i_t = t_offset;
            const Element* f_t = t_offset + M * N;
            const Element* o_t = t_offset + 2 * M * N;
            const Element* c_bar_t = t_offset + 3 * M * N;

            lstm_element_wise<<<element_wise_grid_dim, element_wise_block_dim,
                                smem_size>>>(i_t, f_t, o_t, c_bar_t, c_1, c, h,
                                             kMaxThreads, size);
        }
        float time3 = timer.Stop();
        // FIXME(ying): this hotfix is to avoid affecting existing codes that I
        // do not have a thorough tests.
        if (time_ptr) (*time_ptr) += (time1 + time2 + time3);

        CudaCheck(cudaFree(t));

        return time1 + time2 + time3;
    }
};

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteNonFusedLSTMLayer {
    void operator()(const Element* xs, const Element* w, const Element* u,
                    const Element* c_init, const Element* h_init, Element* c,
                    Element* h, const size_t hidden_size,
                    const size_t batch_size, const size_t seq_length,
                    const size_t depth) {
        Element* o1;
        cudaMalloc(&o1,
                   seq_length * 4 * hidden_size * batch_size * sizeof(Element));
        // Step 1. W@xs
        // TODO: W@xs = o1
        // TODO: Rewrite M, N, K
        // M: 4 * hidden_size
        // N: batch_size * seq_length
        // K: hidden_size
        // WholeShape: [4 * hidden_size, batch_size * seq_length, hidden_size]
        using CuteLSTMGemm =
            CuteLSTMGemm<Element, InstructionShape, ValueMnk, WarpArrangement,
                         CtaTileShape, WholeShape>;
        CuteLSTMGemm cute_lstm_gemm;
        cute_lstm_gemm(w, xs, o1);

        // Setp 2. U@h_init
        Element* o2;
        cudaMalloc(&o2, 4 * hidden_size * batch_size * sizeof(Element));
        // TODO: U@h_init = o2
        // TODO: Rewrite M, N, K
        // WholeShape: [4 * hidden_size, batch_size, hidden_size]
        cute_lstm_gemm(u, h_init, o2);

        // Setp 3. (c_0, h_0) = Activation(o1 + o2)
        // TODO: Element Wise Fused Activation

        // Step 4. for loop seq_length
        for (size_t i = 1; i < seq_length; ++i) {
            // Step 4.1 U@h
            Element* h_i = h + i * hidden_size * batch_size;
            Element* c_i = c + i * hidden_size * batch_size;

            const Element* h_i_1 = h + (i - 1) * hidden_size * batch_size;
            const Element* c_i_1 = c + (i - 1) * hidden_size * batch_size;

            Element* o3;
            cudaMalloc(&o3, 4 * hidden_size * batch_size * sizeof(Element));

            // TODO: U@h_i_1 = o3

            const Element* o4 = o1 + i * 4 * hidden_size * batch_size;

            // Step 4.2. (c_i, h_i) = Activation(o3 + o4)
            // TODO: Element Wise Fused Activation
        }
    }
};

}  // namespace kaleido::core::cuda_kernel
