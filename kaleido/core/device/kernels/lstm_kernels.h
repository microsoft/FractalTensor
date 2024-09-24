#pragma once
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/kernels/gemm_utils.h"
#include "kaleido/core/device/kernels/lstm/inner.h"
#include "kaleido/core/layout.h"
#include "kaleido/core/tile_shape.h"

#include <cute/pointer.hpp>
#include <cute/tensor.hpp>

using namespace cute;

namespace kaleido::core::cuda_kernel {

// i_t = sigmod(W_i@x_t + U_i@h_t_1 + b_i)
// f_t = sigmod(W_f@x_t + U_f@h_t_1 + b_f)
// o_t = sigmod(W_o@x_t + U_o@h_t_1 + b_o)
// c_t = tanh(W_c@x_t + U_c@h_t_1 + b_c)
// W: [W_i, W_f, W_o, W_c]
// U: [U_i, U_f, U_o, U_c]
template <typename Element, typename KeTraits>
__global__ void KeCuteLSTMGate(const Element* W, const Element* U,
                               const Element* x_t, const Element* h_t_1,
                               Element* O) {
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
    Element* gW_ptr = const_cast<Element*>(W) + blockIdx.x * kK * kTM;
    Element* gx_ptr = const_cast<Element*>(x_t) + blockIdx.y * kK * kTN;
    Element* gU_ptr = const_cast<Element*>(U) + blockIdx.x * kK * kTM;
    Element* gh_ptr = const_cast<Element*>(h_t_1) + blockIdx.y * kK * kTN;
    Element* gO_ptr = O + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    int total_block_x = gridDim.x;
    int current_block_x = blockIdx.x;

    // pointers to shared memory tiles
    Element* sW_ptr = shm;
    Element* sx_ptr = shm + kTM * kTK;
    Element* sU_ptr = shm + kTM * kTK + kTK * kTN;
    Element* sh_ptr = shm + kTM * kTK + kTK * kTN + kTM * kTK;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sW;
    typename KeTraits::LoadB_G2S sx;
    typename KeTraits::LoadC_G2S sU;
    typename KeTraits::LoadD_G2S sh;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    auto rW = make_s2rA(sW_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rx = make_s2rB(sx_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto rU = make_s2rA(sU_ptr, tid, typename KeTraits::SmemLayoutC{}, mma);
    auto rh = make_s2rB(sh_ptr, tid, typename KeTraits::SmemLayoutD{}, mma);

    auto acc1 = get_acc<kTM, kTN>(mma);
    auto acc2 = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreE_R2S sO;  // declare register to shared store
    typename KeTraits::StoreE_S2G gO;  // declare shm to global store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        sW.copy(gW_ptr, sW_ptr, tid);
        sx.copy(gx_ptr, sx_ptr, tid);
        sU.copy(gU_ptr, sU_ptr, tid);
        sh.copy(gh_ptr, sh_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rW.get_iters(); ++i) {
            rW.copy(i);
            rx.copy(i);
            gemm(mma, rW[i], rx[i], acc1);
        }

        for (int i = 0; i < rU.get_iters(); ++i) {
            rU.copy(i);
            rh.copy(i);
            gemm(mma, rU[i], rh[i], acc2);
        }

        __syncthreads();

        gW_ptr += kTK;
        gx_ptr += kTK;

        gU_ptr += kTK;
        gh_ptr += kTK;
    }

    axpby(1.0, acc1, 1.0, acc2);  // E = A@B + C@D
    __syncthreads();

    if (current_block_x < total_block_x * 3 / 4) {
        cute_sigmod(acc2);
    } else {
        cute_tanh(acc2);
    }

    __syncthreads();

    sO.copy(acc2, shm, tid);  // store RF tile to shared memory
    __syncthreads();

    gO.copy(shm, gO_ptr,
            tid);  // store shared memory tile to global memory
}

template <typename Element, typename KeTraits>
__global__ void KeCuteBatchedLSTMGate(const Element* W, const Element* U,
                                      const Element* x_t, const Element* h_t_1,
                                      Element* O, int stride_a, int stride_b,
                                      int stride_c, int m, int n, int k,
                                      int b) {
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
    Element* gx_ptr = const_cast<Element*>(x_t) + blockIdx.y * kK * kTN +
                      blockIdx.z * stride_a;
    Element* gh_ptr = const_cast<Element*>(h_t_1) + blockIdx.y * kK * kTN +
                      blockIdx.z * stride_a;

    Element* gW_ptr =
        const_cast<Element*>(W) + blockIdx.x * kK * kTM + blockIdx.z * stride_b;
    Element* gU_ptr =
        const_cast<Element*>(U) + blockIdx.x * kK * kTM + blockIdx.z * stride_b;

    Element* gO_ptr =
        O + blockIdx.x * kTM * kN + blockIdx.y * kTN + blockIdx.z * stride_c;

    int total_block_x = gridDim.x;
    int current_block_x = blockIdx.x;

    // pointers to shared memory tiles
    Element* sW_ptr = shm;
    Element* sx_ptr = shm + kTM * kTK;
    Element* sU_ptr = shm + kTM * kTK + kTK * kTN;
    Element* sh_ptr = shm + kTM * kTK + kTK * kTN + kTM * kTK;

    // declare global to shared memory copy plan
    typename KeTraits::LoadA_G2S sW;
    typename KeTraits::LoadB_G2S sx;
    typename KeTraits::LoadC_G2S sU;
    typename KeTraits::LoadD_G2S sh;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    auto rW = make_s2rA(sW_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rx = make_s2rB(sx_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto rU = make_s2rA(sU_ptr, tid, typename KeTraits::SmemLayoutC{}, mma);
    auto rh = make_s2rB(sh_ptr, tid, typename KeTraits::SmemLayoutD{}, mma);

    auto acc1 = get_acc<kTM, kTN>(mma);
    auto acc2 = get_acc<kTM, kTN>(mma);

    typename KeTraits::StoreE_R2S sO;  // declare register to shared store
    typename KeTraits::StoreE_S2G gO;  // declare shm to global store plan

    for (int k = 0; k < kK; k += kTK) {  // iterator over K
        sW.copy(gW_ptr, sW_ptr, tid);
        sx.copy(gx_ptr, sx_ptr, tid);
        sU.copy(gU_ptr, sU_ptr, tid);
        sh.copy(gh_ptr, sh_ptr, tid);
        __copy_async();
        __syncthreads();

        for (int i = 0; i < rW.get_iters(); ++i) {
            rW.copy(i);
            rx.copy(i);
            gemm(mma, rW[i], rx[i], acc1);
        }

        for (int i = 0; i < rU.get_iters(); ++i) {
            rU.copy(i);
            rh.copy(i);
            gemm(mma, rU[i], rh[i], acc2);
        }

        __syncthreads();

        gW_ptr += kTK;
        gx_ptr += kTK;

        gU_ptr += kTK;
        gh_ptr += kTK;
    }

    axpby(1.0, acc1, 1.0, acc2);  // E = A@B + C@D
    __syncthreads();

    if (current_block_x < total_block_x * 3 / 4) {
        cute_sigmod(acc2);
    } else {
        cute_tanh(acc2);
    }

    __syncthreads();

    sO.copy(acc2, shm, tid);  // store RF tile to shared memory
    __syncthreads();

    gO.copy(shm, gO_ptr,
            tid);  // store shared memory tile to global memory
}

template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
__device__ void copy_tensor_g2s(const Element* src_data, Element* dst_data,
                                SrcLayout src_layout, DstLayout dst_layout,
                                TiledCopy tiled_copy, int tid) {
    auto gtile = make_tensor(make_gmem_ptr(src_data), src_layout);
    auto stile = make_tensor(make_smem_ptr(dst_data), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(gtile);
    auto dst = loader.partition_D(stile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst(_, i, j));
}

template <typename Element, typename SrcLayout, typename DstLayout,
          typename TiledCopy>
__device__ void copy_tensor_s2g(const Element* src_data, Element* dst_data,
                                SrcLayout src_layout, DstLayout dst_layout,
                                TiledCopy tiled_copy, int tid) {
    auto stile = make_tensor(make_smem_ptr(src_data), src_layout);
    auto gtile = make_tensor(make_gmem_ptr(dst_data), dst_layout);

    auto loader = tiled_copy.get_thread_slice(tid);

    auto src = loader.partition_S(stile);
    auto dst = loader.partition_D(gtile);

#pragma unroll
    for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
        for (int j = 0; j < int(size<2>(src)); ++j)
            cute::copy(tiled_copy, src(_, i, j), dst(_, i, j));
}

template <typename Element, typename KeTraits>
__global__ void KeCuteDynamicLstmGate(const Element* ws, const Element* us,
                                      const Element* xs, const Element* hs,
                                      Element* ts, const int m, const int n,
                                      const int k) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    const int kM = m;
    const int kN = n;
    const int kK = k;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    int tid = threadIdx.x;

    // Advance to the global data tile to the current CTA.
    Element* gxs_ptr = const_cast<Element*>(xs) + blockIdx.y * kK * kTN;
    Element* ghs_ptr = const_cast<Element*>(hs) + blockIdx.y * kK * kTN;
    Element* gws_ptr = const_cast<Element*>(ws) + blockIdx.x * kK * kTM;
    Element* gus_ptr = const_cast<Element*>(us) + blockIdx.x * kK * kTM;

    Element* gts_ptr = ts + blockIdx.x * kTM * kN + blockIdx.y * kTN;

    int total_block_x = gridDim.x;
    int current_block_x = blockIdx.x;

    // pointers to shared memory tiles
    Element* sws_ptr = shm;
    Element* sxs_ptr = shm + kTM * kTK;
    Element* sus_ptr = shm + kTM * kTK + kTK * kTN;
    Element* shs_ptr = shm + kTM * kTK + kTK * kTN + kTM * kTK;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    typename KeTraits::TiledCopy tiled_copy;

    auto rws = make_s2rA(sws_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rxs = make_s2rB(sxs_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto rus = make_s2rA(sus_ptr, tid, typename KeTraits::SmemLayoutC{}, mma);
    auto rhs = make_s2rB(shs_ptr, tid, typename KeTraits::SmemLayoutD{}, mma);

    auto acc1 = get_acc<kTM, kTN>(mma);
    auto acc2 = get_acc<kTM, kTN>(mma);

    auto load_a_g2s_layout = make_row_major_layout(kTM, kTK, kK);
    auto load_b_g2s_layout = make_row_major_layout(kTN, kTK, kK);
    auto load_c_g2s_layout = make_row_major_layout(kTM, kTK, kK);
    auto load_d_g2s_layout = make_row_major_layout(kTN, kTK, kK);
    auto store_e_s2g_layout = make_row_major_layout(kTM, kTN, kN);

    typename KeTraits::StoreE_R2S sts;  // declare register to shared store

    for (int k = 0; k < kK; k += kTK) {
        // TODO: Load data from global memory to shared memory
        copy_tensor_g2s(gws_ptr, sws_ptr, load_a_g2s_layout,
                        typename KeTraits::SmemLayoutA{}, tiled_copy, tid);
        copy_tensor_g2s(gxs_ptr, sxs_ptr, load_b_g2s_layout,
                        typename KeTraits::SmemLayoutB{}, tiled_copy, tid);
        copy_tensor_g2s(gus_ptr, sus_ptr, load_c_g2s_layout,
                        typename KeTraits::SmemLayoutC{}, tiled_copy, tid);
        copy_tensor_g2s(ghs_ptr, shs_ptr, load_d_g2s_layout,
                        typename KeTraits::SmemLayoutD{}, tiled_copy, tid);

        __copy_async();
        __syncthreads();

        for (int i = 0; i < rws.get_iters(); i++) {
            rws.copy(i);
            rxs.copy(i);
            gemm(mma, rws[i], rxs[i], acc1);
        }

        for (int i = 0; i < rus.get_iters(); i++) {
            rus.copy(i);
            rhs.copy(i);
            gemm(mma, rus[i], rhs[i], acc2);
        }

        __syncthreads();
        gws_ptr += kTK;
        gxs_ptr += kTK;
        gus_ptr += kTK;
        ghs_ptr += kTK;
    }

    __syncthreads();

    sts.copy(acc1, shm, tid);

    __syncthreads();

    copy_tensor_s2g(shs_ptr, ghs_ptr, typename KeTraits::SmemLayoutE{},
                    store_e_s2g_layout, tiled_copy, tid);
}

template <typename Element, typename KeTraits>
__global__ void KeCuteDynamicBatchedLstmGate(
    const Element* ws, const Element* us, const Element* xs, const Element* hs,
    Element* ts, int stride_a, int stride_b, int stride_c, int m, int n, int k,
    int b) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<Element*>(shared_buf);

    // Whole GEMM shape
    const int kM = m;
    const int kN = n;
    const int kK = k;
    const int kB = b;

    // CTA GEMM shape
    const int kTM = KeTraits::kTM;
    const int kTN = KeTraits::kTN;
    const int kTK = KeTraits::kTK;

    int tid = threadIdx.x;

    // Advance to the global data tile to the current CTA.
    Element* gxs_ptr = const_cast<Element*>(xs) + blockIdx.y * kK * kTN +
                       blockIdx.z * stride_a;
    Element* ghs_ptr = const_cast<Element*>(hs) + blockIdx.y * kK * kTN +
                       blockIdx.z * stride_a;

    Element* gws_ptr = const_cast<Element*>(ws) + blockIdx.x * kK * kTM +
                       blockIdx.z * stride_b;
    Element* gus_ptr = const_cast<Element*>(us) + blockIdx.x * kK * kTM +
                       blockIdx.z * stride_b;

    Element* gts_ptr =
        ts + blockIdx.x * kTM * kN + blockIdx.y * kTN + blockIdx.z * stride_c;

    int total_block_x = gridDim.x;
    int current_block_x = blockIdx.x;

    // pointers to shared memory tiles
    Element* sws_ptr = shm;
    Element* sxs_ptr = shm + kTM * kTK;
    Element* sus_ptr = shm + kTM * kTK + kTK * kTN;
    Element* shs_ptr = shm + kTM * kTK + kTK * kTN + kTM * kTK;

    // declare shared memory to register file copy plan.
    // tcu's wmma instruction prescribes a strict data to thread
    // mapping, in the current implementation, the shm-2-reg copy
    // plan is related to mma.
    typename KeTraits::TiledMma mma;
    typename KeTraits::TiledCopy tiled_copy;

    auto rws = make_s2rA(sws_ptr, tid, typename KeTraits::SmemLayoutA{}, mma);
    auto rxs = make_s2rB(sxs_ptr, tid, typename KeTraits::SmemLayoutB{}, mma);
    auto rus = make_s2rA(sus_ptr, tid, typename KeTraits::SmemLayoutC{}, mma);
    auto rhs = make_s2rB(shs_ptr, tid, typename KeTraits::SmemLayoutD{}, mma);

    auto acc1 = get_acc<kTM, kTN>(mma);
    auto acc2 = get_acc<kTM, kTN>(mma);

    auto load_a_g2s_layout = make_row_major_layout(kTM, kTK, kK);
    auto load_b_g2s_layout = make_row_major_layout(kTN, kTK, kK);
    auto load_c_g2s_layout = make_row_major_layout(kTM, kTK, kK);
    auto load_d_g2s_layout = make_row_major_layout(kTN, kTK, kK);
    auto store_e_s2g_layout = make_row_major_layout(kTM, kTN, kN);

    typename KeTraits::StoreE_R2S sts;  // declare register to shared store

    for (int k = 0; k < kK; k += kTK) {
        // TODO: Load data from global memory to shared memory
        copy_tensor_g2s(gws_ptr, sws_ptr, load_a_g2s_layout,
                        typename KeTraits::SmemLayoutA{}, tiled_copy, tid);
        copy_tensor_g2s(gxs_ptr, sxs_ptr, load_b_g2s_layout,
                        typename KeTraits::SmemLayoutB{}, tiled_copy, tid);
        copy_tensor_g2s(gus_ptr, sus_ptr, load_c_g2s_layout,
                        typename KeTraits::SmemLayoutC{}, tiled_copy, tid);
        copy_tensor_g2s(ghs_ptr, shs_ptr, load_d_g2s_layout,
                        typename KeTraits::SmemLayoutD{}, tiled_copy, tid);

        __copy_async();
        __syncthreads();

        for (int i = 0; i < rws.get_iters(); i++) {
            rws.copy(i);
            rxs.copy(i);
            gemm(mma, rws[i], rxs[i], acc1);
        }

        for (int i = 0; i < rus.get_iters(); i++) {
            rus.copy(i);
            rhs.copy(i);
            gemm(mma, rus[i], rhs[i], acc2);
        }

        __syncthreads();
        gws_ptr += kTK;
        gxs_ptr += kTK;
        gus_ptr += kTK;
        ghs_ptr += kTK;
    }

    __syncthreads();

    sts.copy(acc1, shm, tid);

    __syncthreads();

    copy_tensor_s2g(shs_ptr, ghs_ptr, typename KeTraits::SmemLayoutE{},
                    store_e_s2g_layout, tiled_copy, tid);
}

}  // namespace kaleido::core::cuda_kernel
