#pragma once
#include "kaleido/core/device/kernels/tiled_copy.h"
#include "utils/elementwise.h"
#include "utils/gmem_copy.h"
#include "utils/matmul.h"
#include "utils/misc.h"
#include "utils/reduce.h"

using namespace kaleido::core::cuda_kernel;
using namespace cute;

template <typename InType, typename AccType, typename Traits, int Kd, int D,
          int kTileSizeRow, int kTileSizeCol, int Nthreads, int BlockKSmem = Kd,
          int num_stages_qk = 1, bool load_q_once = true,
          int BlockKSmem2 = kTileSizeCol, int num_stages_v = 1,
          int SmemKAtom = 64, int kSwizzle = 3, bool unrollLastIter = false>
__global__ void __launch_bounds__(Nthreads)
    flashattn(InType* dQ, InType* dK, InType* dV, InType* dO, int length_k,
              int length_q) {
    constexpr float softmax_scale = 1.250000e-01f;

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    InType* shared = reinterpret_cast<InType*>(shared_buf);
    InType* shared_k = shared + num_stages_qk * kTileSizeRow * BlockKSmem;
    InType* shared_v = shared_k + kTileSizeCol * BlockKSmem * num_stages_qk;

    int Tr = length_q / kTileSizeRow;
    int len = length_k;
    int iters = (len + kTileSizeCol - 1) / kTileSizeCol;

    int q_offset = (int)blockIdx.x * Kd * kTileSizeRow;
    int k_offset = ((int)blockIdx.x / Tr) * Kd * length_k;
    int v_offset = ((int)blockIdx.x / Tr) * D * length_k;
    int o_offset = (int)blockIdx.x * D * kTileSizeRow;

    InType* q_ptr = dQ + q_offset;
    InType* k_ptr = dK + k_offset;

    typename Traits::TiledMma tiled_mma;
    typename Traits::CopyG2S g2s_copy;
    auto g2s_thrd_copy = g2s_copy.get_thread_slice(threadIdx.x);

    Tensor gQ = make_tensor(make_gmem_ptr(q_ptr),
                            Shape<Int<kTileSizeRow>, Int<BlockKSmem>>{},
                            make_stride(Int<Kd>{}, _1{}));
    Tensor gQs = g2s_thrd_copy.partition_S(gQ);

    Tensor sQ =
        make_tensor(make_smem_ptr(shared), typename Traits::SmemLayoutQ{});
    Tensor sQs = g2s_thrd_copy.partition_D(sQ);

    Tensor gK = make_tensor(make_gmem_ptr(k_ptr),
                            Shape<Int<kTileSizeCol>, Int<BlockKSmem>>{},
                            make_stride(Int<Kd>{}, _1{}));
    Tensor gKs = g2s_thrd_copy.partition_S(gK);

    Tensor sK = make_tensor(shared_k, typename Traits::SmemLayoutK{});
    Tensor sKs = g2s_thrd_copy.partition_D(sK);

    CopyTilesG2S g2s_qk(g2s_copy, gQs, sQs, gKs, sKs, BlockKSmem, size(sQ),
                        BlockKSmem, size(sK), num_stages_qk);

    CopyTileG2S g2s_k(g2s_copy, gKs, sKs, BlockKSmem, size(sK), num_stages_qk);

    Tensor gV = make_tensor(make_gmem_ptr(dV + v_offset),
                            Shape<Int<BlockKSmem2>, Int<D>>{},
                            make_stride(Int<D>{}, _1{}));
    Tensor gVs = g2s_thrd_copy.partition_S(gV);

    Tensor sV = make_tensor(shared_v, typename Traits::SmemLayoutV{});
    Tensor sVs = g2s_thrd_copy.partition_D(sV);

    CopyTileG2S g2s_v(g2s_copy, gVs, sVs, BlockKSmem2 * D, size(sV),
                      num_stages_v);

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor rQ_org = thr_mma.partition_fragment_A(sQ);
    Tensor rK_org = thr_mma.partition_fragment_B(sK);

    Tensor acc_o =
        partition_fragment_C(tiled_mma, Shape<Int<kTileSizeRow>, Int<D>>{});
    Tensor acc_s = partition_fragment_C(
        tiled_mma, Shape<Int<kTileSizeRow>, Int<kTileSizeCol>>{});

    auto s2r_copyQ =
        make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto s2r_thr_copyQ = s2r_copyQ.get_thread_slice(threadIdx.x);

    Tensor sQs_copy = s2r_thr_copyQ.partition_S(sQ);
    auto smem_tiled_copy_K =
        make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(threadIdx.x);

    Tensor sKs_copy = smem_thr_copy_K.partition_S(sK);
    Tensor sVt =
        make_tensor(sV.data(), typename Traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(
        sVt.data(), typename Traits::SmemLayoutVtransposedNoSwizzle{});

    Tensor rQ = s2r_thr_copyQ.retile_D(rQ_org);
    Tensor rK = smem_thr_copy_K.retile_D(rK_org);

    auto s2r_copyV =
        make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto thr_copy_rV = s2r_copyV.get_thread_slice(threadIdx.x);

    Tensor sVst_copy = thr_copy_rV.partition_S(sVt);
    Tensor rVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor rVt_copy_view = thr_copy_rV.retile_D(rVt);

    CopyTilesS2R qk_s2r(s2r_copyQ, sQs_copy, rQ, smem_tiled_copy_K, sKs_copy,
                        rK, tiled_mma, rQ_org, rK_org, acc_s, size(sQ),
                        size(sK), num_stages_qk);

    CopyTileS2R v_s2r(s2r_copyV, sVst_copy, rVt_copy_view, tiled_mma, rVt,
                      acc_o, size(sV), num_stages_v);

    g2s_qk.prologue();

    Tensor m_new = make_tensor<float>(Shape<Int<2 * size<1>(acc_s)>>{});
    Tensor lse_new = make_fragment_like(m_new);
    fill(lse_new, 0.0f);
    fill(m_new, -INFINITY);
    clear(acc_o);

    for (int i = 0; i < iters - (unrollLastIter ? 1 : 0); ++i) {
        clear(acc_s);
        for (int ax0 = 0; ax0 < Kd / BlockKSmem - 1; ++ax0) {
            cp_async_wait_flash<0>();
            __syncthreads();
            g2s_qk.body();
            qk_s2r.body();
        }
        cp_async_wait_flash<0>();
        __syncthreads();
        g2s_v.prologue();
        qk_s2r.epilogue();

        auto scores =
            make_tensor(acc_s.data(), convert_layout_scores(acc_s.layout()));

        Tensor m_old = make_fragment_like(m_new);
        copy(m_new, m_old);

        Tensor scores_max = make_fragment_like(m_new);
        reduce_max<4, true>(scores, scores_max);

        for (int ax0 = 0; ax0 < size<0>(m_new); ++ax0)
            m_new(ax0) = max(m_new(ax0), scores_max(ax0));

        auto acc_o_rowcol =
            make_tensor(acc_o.data(), convert_layout_scores(acc_o.layout()));

        for (int ax0 = 0; ax0 < size<0>(acc_o_rowcol); ++ax0) {
            float scale = exp((m_old(ax0) - m_new(ax0)) * softmax_scale);
            lse_new(ax0) = lse_new(ax0) * scale;
            for (int ax1 = 0; ax1 < size<1>(acc_o_rowcol); ax1++) {
                acc_o_rowcol(ax0, ax1) *= scale;
            }
        }

        for (int ax0 = 0; ax0 < size<0>(scores); ++ax0) {
            float m_scaled = m_new(ax0) * softmax_scale;
            for (int ax1 = 0; ax1 < size<1>(scores); ax1++) {
                scores(ax0, ax1) =
                    exp(scores(ax0, ax1) * softmax_scale - m_scaled);
            }
        }

        Tensor scores_sum = make_fragment_like(lse_new);
        reduce_sum<4>(scores, scores_sum);

        for (int ax0 = 0; ax0 < size<0>(lse_new); ++ax0) {
            lse_new(ax0) = lse_new(ax0) + scores_sum(ax0);
        }

        auto frag = convert_type<InType>(scores);

        Tensor rP = make_tensor(make_rmem_ptr<InType>(&frag), scores.layout());
        Tensor rP_Aregs =
            make_tensor(rP.data(), convert_layout_rowcol_Aregs(rP.layout()));

        for (int ax0 = 0; ax0 < kTileSizeCol / BlockKSmem2 - 1; ++ax0) {
            cp_async_wait_flash<0>();
            __syncthreads();
            g2s_v.body();
            v_s2r.body(rP_Aregs);
        }

        cp_async_wait_flash<0>();
        __syncthreads();
        if (i < iters - 1) {
            gKs.data() = gKs.data() + (-Kd) + kTileSizeCol * Kd;
            if (load_q_once) {
                g2s_k.prologue();
            } else {
                gQs.data() = gQs.data() + (-Kd);
                g2s_qk.prologue();
            }
        }
        v_s2r.epilogue(rP_Aregs);
    }

    if (unrollLastIter) {
        clear(acc_s);
        for (int ax0 = 0; ax0 < Kd / BlockKSmem - 1; ++ax0) {
            cp_async_wait_flash<0>();
            __syncthreads();
            g2s_qk.body();
            qk_s2r.body();
        }

        cp_async_wait_flash<0>();
        __syncthreads();
        g2s_v.prologue();
        qk_s2r.epilogue();

        Tensor scores =
            make_tensor(acc_s.data(), convert_layout_scores(acc_s.layout()));

        Tensor m_old = make_fragment_like(m_new);
        copy(m_new, m_old);

        Tensor scores_max = make_fragment_like(m_new);
        reduce_max<4, true>(scores, scores_max);

        for (int ax0 = 0; ax0 < size<0>(m_new); ++ax0) {
            m_new(ax0) = max(m_new(ax0), scores_max(ax0));
        }

        Tensor acc_o_rowcol =
            make_tensor(acc_o.data(), convert_layout_scores(acc_o.layout()));

        for (int ax0 = 0; ax0 < size<0>(acc_o_rowcol); ++ax0) {
            float scale = exp((m_old(ax0) - m_new(ax0)) * softmax_scale);
            lse_new(ax0) = lse_new(ax0) * scale;
            for (int ax1 = 0; ax1 < size<1>(acc_o_rowcol); ++ax1) {
                acc_o_rowcol(ax0, ax1) *= scale;
            }
        }

        for (int ax0 = 0; ax0 < size<0>(scores); ++ax0) {
            float m_scaled = m_new(ax0) * softmax_scale;
            for (int ax1 = 0; ax1 < size<1>(scores); ++ax1) {
                scores(ax0, ax1) =
                    exp(scores(ax0, ax1) * softmax_scale - m_scaled);
            }
        }

        Tensor scores_sum = make_fragment_like(lse_new);
        reduce_sum<4>(scores, scores_sum);
        for (int ax0 = 0; ax0 < size<0>(lse_new); ++ax0) {
            lse_new(ax0) = lse_new(ax0) + scores_sum(ax0);
        }

        auto frag = convert_type<InType>(scores);

        Tensor rP = make_tensor(make_rmem_ptr<InType>(&frag), scores.layout());
        Tensor rP_Aregs =
            make_tensor(rP.data(), convert_layout_rowcol_Aregs(rP.layout()));

        for (int ax0 = 0; ax0 < kTileSizeCol / BlockKSmem2 - 1; ++ax0) {
            cp_async_wait_flash<0>();
            __syncthreads();
            g2s_v.body();
            v_s2r.body(rP_Aregs);
        }

        cp_async_wait_flash<0>();
        __syncthreads();
        v_s2r.epilogue(rP_Aregs);
    }

    Tensor acc_o_rowcol =
        make_tensor(acc_o.data(), convert_layout_scores(acc_o.layout()));
    for (int ax0 = 0; ax0 < size<0>(acc_o_rowcol); ++ax0) {
        float scale = 1 / lse_new(ax0);
        lse_new(ax0) = m_new(ax0) * softmax_scale + log(lse_new(ax0));
        for (int ax1 = 0; ax1 < size<1>(acc_o_rowcol); ++ax1) {
            acc_o_rowcol(ax0, ax1) *= scale;
        }
    }

    auto frag2 = convert_type<InType>(acc_o);
    Tensor acc_o_f16 =
        make_tensor(make_rmem_ptr<InType>(&frag2), acc_o.layout());

    Tensor sO = make_tensor(make_smem_ptr((InType*)(shared)),
                            typename Traits::SmemLayoutO{});
    auto smem_tiled_copy_O =
        make_tiled_copy_C(typename Traits::SmemCopyAtomO{}, tiled_mma);

    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(threadIdx.x);
    Tensor sOs = smem_thr_copy_O.partition_D(sO);
    Tensor rO_copy_view = smem_thr_copy_O.retile_S(acc_o_f16);

    Tensor gO = make_tensor(make_gmem_ptr(dO + o_offset),
                            Shape<Int<kTileSizeRow>, Int<D>>{},
                            make_stride(Int<D>{}, _1{}));

    typename Traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(threadIdx.x);

    Tensor gO_partition = gmem_thr_copy_O.partition_D(gO);
    Tensor sO_partition = gmem_thr_copy_O.partition_S(sO);
    __syncthreads();
    copy(smem_tiled_copy_O, rO_copy_view, sOs);
    __syncthreads();

    for (int m = 0; m < size<1>(gO_partition); ++m) {
        for (int k = 0; k < size<2>(gO_partition); ++k) {
            copy(gmem_tiled_copy_O, sO_partition(_, m, k),
                 gO_partition(_, m, k));
        }
    }
}
