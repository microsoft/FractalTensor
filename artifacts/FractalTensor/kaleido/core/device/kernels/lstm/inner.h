#pragma once
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/kernels/lstm_kernel_traits.h"

namespace kaleido::core::cuda_kernel {

template <class XEngine, class XLayout>
CUTE_HOST_DEVICE void cute_tanh(cute::Tensor<XEngine, XLayout>& tensor) {
    CUTE_UNROLL
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = tanh(tensor(i));
    }
}

template <class XEngine, class XLayout>
CUTE_HOST_DEVICE void cute_sigmod(cute::Tensor<XEngine, XLayout>& tensor) {
    CUTE_UNROLL
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = 1.0 / (1.0 + exp(-tensor(i)));
    }
}

template <typename Element, typename KeTraits>
__global__ void KeCute2GemmAdd(const Element* dA, const Element* dB,
                               const Element* dC, const Element* dD,
                               Element* dE) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Whole GEMM shape
    const int kM = KeTraits::kM;
    const int kN = KeTraits::kN;
    const int kK = KeTraits::kK;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    int tid = threadIdx.x;

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
    Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
    Element* gC_ptr = const_cast<Element*>(dC) + blockIdx.x * kK * kTM;
    Element* gD_ptr = const_cast<Element*>(dD) + blockIdx.y * kK * kTN;
    Element* gE_ptr = dE + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;
    Element* sC_ptr = shm + kTM * kTK + kTK * kTN;
    Element* sD_ptr = shm + kTM * kTK + kTK * kTN + kTM * kTK;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sA;
    typename KeTraits::LoadB_G2S sB;
    typename KeTraits::LoadC_G2S sC;
    typename KeTraits::LoadD_G2S sD;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto rC = make_s2rA(sC_ptr, tid, typename KeTraits::SmemLayoutC{}, mma);
    auto rD = make_s2rB(sD_ptr, tid, typename KeTraits::SmemLayoutD{}, mma);

    auto acc1 = get_acc<kTM, kTN>(mma);
    auto acc2 = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreE_R2S sE;  // declare register to shared store plan
    typename KeTraits::StoreE_S2G gE;  // declare shm to global store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        sA.copy(gA_ptr, sA_ptr, tid);
        sB.copy(gB_ptr, sB_ptr, tid);
        sC.copy(gC_ptr, sC_ptr, tid);
        sD.copy(gD_ptr, sD_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            gemm(mma, rA[i], rB[i],
                 acc1);  // compute using tcu's wmma instruction
        }

        for (int i = 0; i < rC.get_iters(); ++i) {
            rC.copy(i);  // load C register tile from shared memory
            rD.copy(i);  // load D register tile from shared memory

            gemm(mma, rC[i], rD[i],
                 acc2);  // compute using tcu's wmma instruction
        }

        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;

        gC_ptr += kTK;
        gD_ptr += kTK;
    }

    axpby(1.0, acc1, 1.0, acc2);  // E = A@B + C@D
    __syncthreads();

    sE.copy(acc2, shm,
            tid);  // store shared memory tile to global memory
    __syncthreads();

    gE.copy(shm, gE_ptr,
            tid);  // store shared memory tile to global memory
}

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct Cute2GemmAdd {
    void operator()(const Element* A, const Element* B, const Element* C,
                    const Element* D, Element* E) {
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
            GemmAddTraits<Element, InstructionShape, ValueMnk, WarpArrangement,
                          CtaTileShape, WholeShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        auto kernel = &KeCute2GemmAdd<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }

        const int block_m = CeilDiv<kM, kTM>;
        const int block_n = CeilDiv<kN, kTN>;

        const int kThreads = KeTraits::kThreads;

        dim3 gridDim(block_m, block_n);
        dim3 blockDim(kThreads, 1, 1);

        kernel<<<gridDim, blockDim, smem_size>>>(A, B, C, D, E);
    }
};

// sigmod(A@B + C@D)
template <typename Element, typename KeTraits>
__global__ void KeCuteGemmAddSigmod(const Element* dA, const Element* dB,
                                    const Element* dC, const Element* dD,
                                    Element* dE) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Whole GEMM shape
    const int kM = KeTraits::kM;
    const int kN = KeTraits::kN;
    const int kK = KeTraits::kK;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    int tid = threadIdx.x;

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
    Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
    Element* gC_ptr = const_cast<Element*>(dC) + blockIdx.x * kK * kTM;
    Element* gD_ptr = const_cast<Element*>(dD) + blockIdx.y * kK * kTN;
    Element* gE_ptr = dE + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;
    Element* sC_ptr = shm + kTM * kTK + kTK * kTN;
    Element* sD_ptr = shm + kTM * kTK + kTK * kTN + kTM * kTK;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sA;
    typename KeTraits::LoadB_G2S sB;
    typename KeTraits::LoadC_G2S sC;
    typename KeTraits::LoadD_G2S sD;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto rC = make_s2rA(sC_ptr, tid, typename KeTraits::SmemLayoutC{}, mma);
    auto rD = make_s2rB(sD_ptr, tid, typename KeTraits::SmemLayoutD{}, mma);

    auto acc1 = get_acc<kTM, kTN>(mma);
    auto acc2 = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreE_R2S sE;  // declare register to shared store plan
    typename KeTraits::StoreE_S2G gE;  // declare shm to global store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        sA.copy(gA_ptr, sA_ptr, tid);
        sB.copy(gB_ptr, sB_ptr, tid);
        sC.copy(gC_ptr, sC_ptr, tid);
        sD.copy(gD_ptr, sD_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            gemm(mma, rA[i], rB[i],
                 acc1);  // compute using tcu's wmma instruction
        }

        for (int i = 0; i < rC.get_iters(); ++i) {
            rC.copy(i);  // load C register tile from shared memory
            rD.copy(i);  // load D register tile from shared memory

            gemm(mma, rC[i], rD[i],
                 acc2);  // compute using tcu's wmma instruction
        }

        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;

        gC_ptr += kTK;
        gD_ptr += kTK;
    }

    axpby(1.0, acc1, 1.0, acc2);  // E = A@B + C@D
    __syncthreads();

    cute_sigmod(acc2);
    __syncthreads();

    sE.copy(acc2, shm,
            tid);  // store shared memory tile to global memory
    __syncthreads();

    gE.copy(shm, gE_ptr,
            tid);  // store shared memory tile to global memory
}

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteGemmAddSigmod {
    void operator()(const Element* A, const Element* B, const Element* C,
                    const Element* D, Element* E) {
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
            GemmAddSigmodTraits<Element, InstructionShape, ValueMnk,
                                WarpArrangement, CtaTileShape, WholeShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        auto kernel = &KeCuteGemmAddSigmod<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }

        const int block_m = CeilDiv<kM, kTM>;
        const int block_n = CeilDiv<kN, kTN>;

        const int kThreads = KeTraits::kThreads;

        dim3 gridDim(block_m, block_n);
        dim3 blockDim(kThreads, 1, 1);

        kernel<<<gridDim, blockDim, smem_size>>>(A, B, C, D, E);
    }
};

// tanh(A@B + C@D)
template <typename Element, typename KeTraits>
__global__ void KeCuteGemmAddTanh(const Element* dA, const Element* dB,
                                  const Element* dC, const Element* dD,
                                  Element* dE) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Whole GEMM shape
    const int kM = KeTraits::kM;
    const int kN = KeTraits::kN;
    const int kK = KeTraits::kK;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    int tid = threadIdx.x;

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr = const_cast<Element*>(dA) + blockIdx.x * kK * kTM;
    Element* gB_ptr = const_cast<Element*>(dB) + blockIdx.y * kK * kTN;
    Element* gC_ptr = const_cast<Element*>(dC) + blockIdx.x * kK * kTM;
    Element* gD_ptr = const_cast<Element*>(dD) + blockIdx.y * kK * kTN;
    Element* gE_ptr = dE + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;
    Element* sC_ptr = shm + kTM * kTK + kTK * kTN;
    Element* sD_ptr = shm + kTM * kTK + kTK * kTN + kTM * kTK;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sA;
    typename KeTraits::LoadB_G2S sB;
    typename KeTraits::LoadC_G2S sC;
    typename KeTraits::LoadD_G2S sD;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto rC = make_s2rA(sC_ptr, tid, typename KeTraits::SmemLayoutC{}, mma);
    auto rD = make_s2rB(sD_ptr, tid, typename KeTraits::SmemLayoutD{}, mma);

    auto acc1 = get_acc<kTM, kTN>(mma);
    auto acc2 = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreE_R2S sE;  // declare register to shared store plan
    typename KeTraits::StoreE_S2G gE;  // declare shm to global store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        sA.copy(gA_ptr, sA_ptr, tid);
        sB.copy(gB_ptr, sB_ptr, tid);
        sC.copy(gC_ptr, sC_ptr, tid);
        sD.copy(gD_ptr, sD_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            gemm(mma, rA[i], rB[i],
                 acc1);  // compute using tcu's wmma instruction
        }

        for (int i = 0; i < rC.get_iters(); ++i) {
            rC.copy(i);  // load C register tile from shared memory
            rD.copy(i);  // load D register tile from shared memory

            gemm(mma, rC[i], rD[i],
                 acc2);  // compute using tcu's wmma instruction
        }

        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;

        gC_ptr += kTK;
        gD_ptr += kTK;
    }

    axpby(1.0, acc1, 1.0, acc2);  // E = A@B + C@D
    __syncthreads();

    cute_tanh(acc2);
    __syncthreads();

    sE.copy(acc2, shm,
            tid);  // store shared memory tile to global memory
    __syncthreads();

    gE.copy(shm, gE_ptr,
            tid);  // store shared memory tile to global memory
}

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteGemmAddTanh {
    void operator()(const Element* A, const Element* B, const Element* C,
                    const Element* D, Element* E) {
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
            GemmAddTanhTraits<Element, InstructionShape, ValueMnk,
                              WarpArrangement, CtaTileShape, WholeShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        auto kernel = &KeCuteGemmAddTanh<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }

        const int block_m = CeilDiv<kM, kTM>;
        const int block_n = CeilDiv<kN, kTN>;

        const int kThreads = KeTraits::kThreads;

        dim3 gridDim(block_m, block_n);
        dim3 blockDim(kThreads, 1, 1);

        kernel<<<gridDim, blockDim, smem_size>>>(A, B, C, D, E);
    }
};

template <typename Element, typename KeTraits>
__global__ void KeCuteLSTMGemm(const Element* w, const Element* xs,
                               Element* o) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    const int kM = KeTraits::kM;
    const int kN = KeTraits::kN;
    const int kK = KeTraits::kK;

    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    int tid = threadIdx.x;

    const Element* gw_ptr = const_cast<Element*>(w) + blockIdx.x * kK * kTM;
    const Element* gxs_ptr = const_cast<Element*>(xs) + blockIdx.y * kK * kTN;
    const Element* go_ptr = o + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    typename KeTraits::LoadW_G2S sw;
    typename KeTraits::LoadX_G2S sxs;

    Element* sw_ptr = shm;
    Element* sxs_ptr = shm + kTM * kTK;

    typename KeTraits::TiledMma mma;
    auto rw = make_s2rA(sw_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rxs = make_s2rB(sxs_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);

    auto acc = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreO_R2S so;
    typename KeTraits::StoreO_S2G go;

    for (int k = 0; k < kK; k += kTK) {
        sw.copy(gw_ptr, sw_ptr, tid);
        sxs.copy(gxs_ptr, sxs_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rw.get_iters(); ++i) {
            rw.copy(i);
            rxs.copy(i);
            gemm(mma, rw[i], rxs[i], acc);
        }

        __syncthreads();

        gw_ptr += kTK;
        gxs_ptr += kTK;
    }

    __syncthreads();
    so.copy(acc, shm, tid);

    __syncthreads();
    go.copy(shm, go_ptr, tid);
}

template <typename Element, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape>
struct CuteLSTMGemm {
    void operator()(const Element* w, const Element* xs, Element* o) {
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
            GemmAddTraits<Element, InstructionShape, ValueMnk, WarpArrangement,
                          CtaTileShape, WholeShape>;

        static constexpr int smem_size =
            std::max(kTK * (kTN + kTM) * 2, kTM * kTN) * sizeof(Element);

        auto kernel = &KeCuteLSTMGemm<Element, KeTraits>;

        // maximal statically allocated smem per block
        const int kMaxSmemPerBlock = 48 * 1024;
        if (smem_size > kMaxSmemPerBlock) {
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }

        const int block_m = CeilDiv<kM, kTM>;
        const int block_n = CeilDiv<kN, kTN>;

        const int kThreads = KeTraits::kThreads;

        dim3 gridDim(block_m, block_n);
        dim3 blockDim(kThreads, 1, 1);

        kernel<<<gridDim, blockDim, smem_size>>>(w, xs, o);
    }
};

// TODO: More optimization techniques
template <typename Element>
__global__ void KeLSTMElementWise(const Element* I_t, const Element* F_t,
                                  const Element* O_t, const Element* C_t_bar,
                                  const Element* C_t_1, Element* C_t,
                                  Element* H_t, int block_size, int size) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Add Batch Size
    int batch = blockIdx.y;
    const Element* i = I_t + 4 * batch * size;
    const Element* f = F_t + 4 * batch * size;
    const Element* o = O_t + 4 * batch * size;
    const Element* c_bar = C_t_bar + 4 * batch * size;
    const Element* c_1 = C_t_1 + 4 * batch * size;
    Element* c = C_t + 4 * batch * size;
    Element* h = H_t + 4 * batch * size;

    int tid = threadIdx.x;

    int index = blockIdx.x * block_size + tid;

    if (index < size) {
        // TODO: Loading data into shared memory and computing, versus computing
        // directly in global memory, does not seem to make a difference. This
        // seems to require further optimization, such as reconsidering
        // redistributing data to different threads and performing vectorized
        // loading and storing.

        // This is a very naive kernel that loads data into shared memory and
        // then performs computations. It has been temporarily commented out.

        // // f_t
        // shm[6 * tid] = f[index];
        // // c_t_1
        // shm[6 * tid + 1] = c_1[index];
        // // i_t
        // shm[6 * tid + 2] = i[index];
        // // c_t_bar
        // shm[6 * tid + 3] = c_bar[index];
        // // o_t
        // shm[6 * tid + 4] = o[index];

        // // c_t = f_t * c_t_1 + i_t * c_t_bar
        // shm[6 * tid + 5] = shm[6 * tid] * shm[6 * tid + 1] +
        //                    shm[6 * tid + 2] * shm[6 * tid + 3];

        // __syncthreads();

        // // Store c_t
        // c[index] = shm[6 * tid + 5];

        // __syncthreads();

        // // h_t = o_t * tanh(c_t)
        // shm[6 * tid + 5] = shm[6 * tid + 4] * tanh(shm[6 * tid + 5]);

        // __syncthreads();

        // // Store h_t
        // h[index] = shm[6 * tid + 5];

        c[index] = f[index] * c_1[index] + i[index] * c_bar[index];

        __syncthreads();

        h[index] = o[index] * tanh(c[index]);
    }
}

template <typename Element>
__global__ void KeLSTMFusedElementWise(const Element* I_t, const Element* F_t,
                                       const Element* O_t,
                                       const Element* C_t_bar,
                                       const Element* C_t_1, Element* C_t,
                                       Element* H_t, int BLOCK_SIZE, int SIZE) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);
}

}  // namespace kaleido::core::cuda_kernel
