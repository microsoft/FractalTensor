#pragma once
#include "kaleido/core/device/kernels/gemm_kernel_traits.h"
#include "kaleido/core/device/kernels/gemm_utils.h"
#include "kaleido/core/layout.h"
#include "kaleido/core/tile_shape.h"

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>
#include <cutlass/aligned_buffer.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>

namespace kaleido {
namespace core {
namespace cuda_kernel {

// for lecacy reason, cutlass2 gemm kernel,
template <typename Element, typename Mma, typename WholeShape,
          typename ThreadBlockShape, typename WarpShape,
          typename InstructionShape, typename LoadA, typename LoadB,
          typename StoreC, int smem_count>
__global__ void KeGemm(const Element* dA, const Element* dB, Element* dC,
                       LoadA loader_a, LoadB loader_b, StoreC storer_c) {
    // Whole GEMM shape
    const int kM = WholeShape::kM;
    const int kN = WholeShape::kN;
    const int kK = WholeShape::kK;

    // CTA GEMM shape
    const int kTM = ThreadBlockShape::kM;
    const int kTN = ThreadBlockShape::kN;
    const int kTK = ThreadBlockShape::kK;

    // Warp GEMM shape
    const int kWM = WarpShape::kM;
    const int kWN = WarpShape::kN;
    const int kWK = WarpShape::kK;

    const int ctaRow = blockIdx.x;
    const int ctaCol = blockIdx.y;

    // threads in a CTA can be viewed as laied out in a row-major
    // matrix
    const int warpIdx = threadIdx.x / 32;
    const int num_warp_per_col = kTN / kWN;
    const int warpRow = warpIdx / num_warp_per_col;
    const int warpCol = warpIdx % num_warp_per_col;

    // advance the start positions of A, B, C shared memory tiles
    // relative to the input A, B, C matrix
    Element* cta_A = const_cast<Element*>(dA) + ctaRow * kK * kTM;
    Element* cta_B = const_cast<Element*>(dB) + ctaCol * kK * kTN;
    Element* cta_C =
        const_cast<Element*>(dC) + ctaRow * kTM * kN + ctaCol * kTN;

    __shared__ cutlass::AlignedBuffer<Element, smem_count> smem_buf;
    auto* smem_buf_a = smem_buf.data();
    auto* smem_buf_b = smem_buf_a + kTK * kTM;
    auto* smem_buf_c = smem_buf_a;

    using FragmentA = typename Mma::FragmentA;
    using FragmentB = typename Mma::FragmentB;
    using FragmentC = typename Mma::FragmentC;

    Element* warp_A = smem_buf_a + warpRow * kWM * kTK;
    Element* warp_B = smem_buf_b + warpCol * kWN * kTK;

    Mma mma;
    FragmentA frag_A;
    FragmentB frag_B;
    FragmentC accum;
    accum.clear();

    for (int bkIdx = 0; bkIdx < kK; bkIdx += kTK) {
        /// Load data from global memory into shared memory
        loader_a.template transfer(cta_A, smem_buf_a, kK, threadIdx.x);
        loader_b.template transfer(cta_B, smem_buf_b, kK, threadIdx.x);
        __syncthreads();

        /// Load warp tile from shared memory into register
        typename Mma::LayoutA layout_A = Mma::LayoutA::packed({kTM, kTK});
        typename Mma::IteratorA iter_A({warp_A, layout_A},
                                       cutlass::arch::LaneId());
        typename Mma::LayoutB layout_B = Mma::LayoutB::packed({kTK, kTN});
        typename Mma::IteratorB iter_B({warp_B, layout_B},
                                       cutlass::arch::LaneId());

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kTK; k += Mma::Policy::MmaShape::kK) {
            iter_A.load(frag_A);
            iter_B.load(frag_B);

            mma(accum, frag_A, frag_B, accum);

            ++iter_A;
            ++iter_B;
        }
        __syncthreads();

        cta_A += kTK;  // advance pointer along the K dimension
        cta_B += kTK;
    }

    /// store C tile from register into shared memory
    Element* warp_C = smem_buf_c + warpRow * kWM * kTN + warpCol * kWN;
    typename Mma::IteratorC iter_C({warp_C, cutlass::layout::RowMajor(kTN)},
                                   cutlass::arch::LaneId());
    iter_C.store(accum);
    __syncthreads();

    /// store C tile from shared memory into to global memory.
    storer_c.transfer(smem_buf_c, cta_C, kN, threadIdx.x);
}

/// GEMM kernel using cutlass3 APIs
template <typename Element, typename KeTraits>
__global__ void KeCuteGemm(const Element* dA, const Element* dB, Element* dC) {
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
    Element* gC_ptr = dC + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sA;
    typename KeTraits::LoadB_G2S sB;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto acc = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreC_R2S sC;  // declare register to shared store plan
    typename KeTraits::StoreC_S2G gC;  // declare shm to global store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        sA.copy(gA_ptr, sA_ptr, tid);
        sB.copy(gB_ptr, sB_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            gemm(mma, rA[i], rB[i],
                 acc);  // compute using tcu's wmma instruction
        }
        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;
    }

    sC.copy(acc, shm, tid);  // store register tile to shared memory
    __syncthreads();

    gC.copy(shm, gC_ptr,
            tid);  // store shared memory tile to global memory
}

// Batched GEMM kernel using cutlass3 APIs
// (B * M * K) @ (B * K * N) = (B * M * N)
template <typename Element, typename KeTraits>
__global__ void KeBatchedCuteGemm(const Element* dA, const Element* dB,
                                  Element* dC) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Whole GEMM shape
    const int kM = KeTraits::kM;
    const int kN = KeTraits::kN;
    const int kK = KeTraits::kK;
    const int kB = KeTraits::kB;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    int tid = threadIdx.x;

    // Advance to the global data tile to the current CTA.
    Element* gA_ptr =
        const_cast<Element*>(dA) + blockIdx.x * kK * kTM + blockIdx.z * kK * kM;
    Element* gB_ptr =
        const_cast<Element*>(dB) + blockIdx.y * kK * kTN + blockIdx.z * kK * kN;
    Element* gC_ptr =
        dC + blockIdx.x * kTM * kN + blockIdx.y * kTN + blockIdx.z * kM * kN;

    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sA;
    typename KeTraits::LoadB_G2S sB;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto acc = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreC_R2S sC;  // declare register to shared store plan
    typename KeTraits::StoreC_S2G gC;  // declare shm to global store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        sA.copy(gA_ptr, sA_ptr, tid);
        sB.copy(gB_ptr, sB_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rA.get_iters(); ++i) {
            rA.copy(i);  // load A register tile from shared memory
            rB.copy(i);  // load B register tile from shared memory

            gemm(mma, rA[i], rB[i],
                 acc);  // compute using tcu's wmma instruction
        }
        __syncthreads();

        gA_ptr += kTK;
        gB_ptr += kTK;
    }

    sC.copy(acc, shm, tid);  // store register tile to shared memory
    __syncthreads();

    gC.copy(shm, gC_ptr,
            tid);  // store shared memory tile to global memory
}

template <typename Element, typename KeTraits>
__global__ void KeGemmPipelined(const Element* dA, const Element* dB,
                                Element* dC) {
    using namespace cute;

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

    const int num_stages = KeTraits::num_stages;

    int Tr = kM / kTM;  // number of data tiles along the row dimension
    int Tc = kN / kTN;  // number of data tiles along the column dimension

    int a_offset = (int)blockIdx.x % Tr * kK * kTM;
    int b_offset = (int)blockIdx.x / Tr * kK * kTN;
    int c_offset = (int)blockIdx.x / Tr * kTN + (int)blockIdx.x % Tr * kTM * kN;

    // load the first A, B tiles from global memory to shared memory
    typename KeTraits::GmemTiledCopy tiled_copy;
    auto copy_thrd = tiled_copy.get_thread_slice(threadIdx.x);

    auto gA = make_tensor(make_gmem_ptr(dA + a_offset),
                          typename KeTraits::GmemLayoutA{});
    auto gA_thrd = copy_thrd.partition_S(gA);
    auto sA = make_tensor(make_smem_ptr(shm), typename KeTraits::SmemLayoutA{});
    auto sA_thrd = copy_thrd.partition_D(sA);

    auto gB = make_tensor(make_gmem_ptr(dB + b_offset),
                          typename KeTraits::GmemLayoutB{});
    auto gB_thrd = copy_thrd.partition_S(gB);
    auto sB = make_tensor(sA.data() + num_stages * size(sA),
                          typename KeTraits::SmemLayoutB{});
    auto sB_thrd = copy_thrd.partition_D(sB);

    CopyAsyncG2S g2s(num_stages, tiled_copy, gA_thrd, kTK, sA_thrd, size(sA),
                     gB_thrd, kTK, sB_thrd, size(sB));

    g2s.copy();  // commit the 1st async copy group
    g2s.copy();  // commit the 2nd async copy group
    // Allows for one unfinished cp.async operation.
    g2s.template wait_group<1>();
    __syncthreads();

    typename KeTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<kTM>, Int<kTN>>{});
    clear(acc);

    // data tiles that are stored on local registers
    auto s2r_copy_A =
        make_tiled_copy_A(typename KeTraits::SmemCopyAtom{}, tiled_mma);
    auto s2r_copy_A_thrd = s2r_copy_A.get_thread_slice(threadIdx.x);
    auto sArA = s2r_copy_A_thrd.partition_S(sA);
    auto rA = thr_mma.partition_fragment_A(sA);
    auto rA_view = s2r_copy_A_thrd.retile_D(rA);  // retile for copy

    auto s2r_copy_B =
        make_tiled_copy_B(typename KeTraits::SmemCopyAtom{}, tiled_mma);
    auto s2r_copy_B_thrd = s2r_copy_B.get_thread_slice(threadIdx.x);
    auto sBrB = s2r_copy_B_thrd.partition_S(sB);
    auto rB = thr_mma.partition_fragment_B(sB);
    auto rB_view = s2r_copy_B_thrd.retile_D(rB);  // retile for copy

    static_assert(size<2>(rA) == size<2>(rB),
                  "Error partition of thread tiles.");
    const int k_tiles = size<2>(rB);
    const int Na = size<1>(sA_thrd) * size<2>(sA_thrd);
    const int na = CeilDiv<Na, k_tiles>;
    const int stride_a = size<2>(sA_thrd);

    const int Nb = size<1>(sB_thrd) * size<2>(sB_thrd);
    const int nb = CeilDiv<Nb, k_tiles>;
    const int stride_b = size<2>(sB_thrd);

    // issue the first data loading from shared memory to register
    cute::copy(s2r_copy_A, sArA(_, _, _0{}), rA_view(_, _, _0{}));
    cute::copy(s2r_copy_B, sBrB(_, _, _0{}), rB_view(_, _, _0{}));

    // stage 1
    for (int blk = 0; blk < kK / kTK - 2; ++blk) {
        CUTE_UNROLL
        for (int i = 0; i < k_tiles; ++i) {
            // circular issue next data loading from shared memory
            // into registers
            int pos = (i + 1) % k_tiles;
            cute::copy(s2r_copy_A, sArA(_, _, pos), rA_view(_, _, pos));
            cute::copy(s2r_copy_B, sBrB(_, _, pos), rB_view(_, _, pos));

            if (i < k_tiles - 1) {
                // gmem -> shared memory
                g2s.copy2(i, na, Na, stride_a, nb, Nb, stride_b);
            }

            if (i == k_tiles - 2) {
                sArA.data() = sArA.data() + size(sA);
                sBrB.data() = sBrB.data() + size(sB);

                if ((blk + 1) % num_stages == 0) {
                    sArA.data() = sArA.data() + (-size(sA) * num_stages);
                    sBrB.data() = sBrB.data() + (-size(sB) * num_stages);
                }

                g2s.copy2(i + 1, na, Na, stride_a, nb, Nb, stride_b);
                g2s.commit_copy_group();
                g2s.next();
                if ((blk + 2 + 1) % num_stages == 0) g2s.cycle_dst();

                g2s.template wait_group<1>();
                __syncthreads();
            }

            cute::gemm(tiled_mma, rA(_, _, i), rB(_, _, i),
                       acc);  // compute
        }
    }

    // stage 2
    CUTE_UNROLL
    for (int i = 0; i < k_tiles; ++i) {
        // circular issue next data loading from shared memory into
        // registers
        int pos = (i + 1) % k_tiles;
        cute::copy(s2r_copy_A, sArA(_, _, pos), rA_view(_, _, pos));
        cute::copy(s2r_copy_B, sBrB(_, _, pos), rB_view(_, _, pos));

        if (i == k_tiles - 2) {
            sArA.data() = sArA.data() + size(sA);
            sBrB.data() = sBrB.data() + size(sB);
            if ((kK / kTK - 2 + 1) % num_stages == 0) {
                sArA.data() = sArA.data() + (-size(sA) * num_stages);
                sBrB.data() = sBrB.data() + (-size(sB) * num_stages);
            }

            g2s.template wait_group<0>();
            __syncthreads();
        }

        cute::gemm(tiled_mma, rA(_, _, i), rB(_, _, i),
                   acc);  // compute
    }

    // stage 3
    CUTE_UNROLL
    for (int i = 0; i < k_tiles; ++i) {
        if (i < k_tiles - 1) {
            // circular issue next data loading from shared memory
            // into registers
            int pos = (i + 1) % k_tiles;
            cute::copy(s2r_copy_A, sArA(_, _, pos), rA_view(_, _, pos));
            cute::copy(s2r_copy_B, sBrB(_, _, pos), rB_view(_, _, pos));
        }

        cute::gemm(tiled_mma, rA(_, _, i), rB(_, _, i),
                   acc);  // compute
    }
    __syncthreads();

    // convert the accumulator to the output type
    auto rC = convert_type<Element>(acc);
    // store the result from register to shared memory
    auto r2s = make_tiled_copy_C(typename KeTraits::SmemCopyAtomO{}, tiled_mma);
    auto sC = make_tensor(make_smem_ptr(shm), typename KeTraits::SmemLayoutO{});
    R2S_copy(r2s, rC, sC, threadIdx.x);
    __syncthreads();

    // store the result from shared memory to global memory
    auto gC = make_tensor(make_gmem_ptr(dC + c_offset),
                          typename KeTraits::GmemLayoutC{});
    typename KeTraits::GmemTiledCopyO s2g;
    S2G_copy(s2g, sC, gC, threadIdx.x);
}

// D = A @ B @ C
template <typename Element, typename KeTraits>
__global__ void KeBack2BackGemm(const Element* dA, const Element* dB,
                                const Element* dC, Element* dD) {
    // whole problem shape
    const int kM = KeTraits::kM;
    const int kN = KeTraits::kN;
    const int kK = KeTraits::kK;
    const int kP = KeTraits::kP;

    // shared memory tile shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;
    const int kTP = KeTraits::kTP;

    // Advance to the global data tile to the current CTA.
    Element* A = const_cast<Element*>(dA) + blockIdx.x * (kTM * kK);
    Element* B = const_cast<Element*>(dB);
    Element* gC_ptr = const_cast<Element*>(dC) + blockIdx.y * (kTP * kN);
    Element* gD_ptr = dD + blockIdx.x * (kTM * kP) + (blockIdx.y * kTP);

    Element* gA_ptr;
    Element* gB_ptr;

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);
    // pointers to shared memory tiles
    Element* sA_ptr = shm;
    Element* sB_ptr = shm + kTM * kTK;
    Element* sC_ptr = shm + kTM * kTK + kTK * kTN;

    int tid = threadIdx.x;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sA;
    typename KeTraits::LoadB_G2S sB;
    typename KeTraits::LoadC_G2S sC;

    typename KeTraits::TiledMma mma;  // for shared memory to register copy
    auto rA = make_s2rA(sA_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rB = make_s2rB(sB_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto acc1 = get_acc<kTM, kTN>(mma);  // accumulator for the 1st gemm

    auto rC = make_s2rB(sC_ptr, tid, typename KeTraits::SmemLayoutC{}, mma);
    auto acc2 = get_acc<kTM, kTP>(mma);  // accumulator for the 2nd gemm

    typename KeTraits::StoreD_R2S sD;  // declare register to shared store plan
    typename KeTraits::StoreD_S2G gD;  // declare shm to global store plan

    for (int n = 0; n < kN; n += kTN) {  // iterate over N
        gA_ptr = A;                      // A tile is repeated loaded
        gB_ptr = B + n * kK;
        for (int k = 0; k < kK; k += kTK) {  // iterate over K
            sA.copy(gA_ptr, sA_ptr,
                    tid);  // load A tile from global to shared
            sB.copy(gB_ptr, sB_ptr,
                    tid);  // load B tile from global to shared
            __copy_async();
            __syncthreads();

            // iterate over the register tiles along the kTK
            // dimension
            for (int i = 0; i < rA.get_iters(); ++i) {
                rA.copy(i);  // load A register tile from shared memory
                rB.copy(i);  // load B register tile from shared memory
                gemm(mma, rA[i], rB[i], acc1);  // compute
            }
            __syncthreads();

            gA_ptr += kTK;
            gB_ptr += kTK;
        }
        // The output type of the first tensor core matrix
        // multiplication is float32. However, before the second
        // GEMM operation, the output needs to be converted to half
        // precision.
        auto acc_half = convert_type<Element>(acc1);
        auto rA2 = convert_layout<KeTraits::TiledMma>(acc_half);

        // load C tile from global to shared memory
        sC.copy(gC_ptr, sC_ptr, tid);
        __copy_async();
        __syncthreads();

        // iterate over register tiles along the kTN dimension
        for (int i = 0; i < rC.get_iters(); ++i) {
            rC.copy(i);  // load C tile from shared memory to register
            gemm(mma, rA2[i], rC[i], acc2);  // compute
        }
        __syncthreads();

        clear(acc1);
        gC_ptr += kTN;
    }

    sD.copy(acc2, shm,
            tid);  // store register tile to shared memory
    __syncthreads();
    gD.copy(shm, gD_ptr,
            tid);  // store shared memory tile to global memory
}

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
