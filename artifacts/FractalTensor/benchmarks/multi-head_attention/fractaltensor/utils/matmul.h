#pragma once
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

using namespace cute;

template <class SmemTiledCopy1, class STensor1, class RTensor1,
          class SmemTiledCopy2, class STensor2, class RTensor2,
          class TiledMMAType, class RTensor3, class RTensor4, class RTensor5>
class MatmulQK_s2r {
   public:
    __device__ MatmulQK_s2r(SmemTiledCopy1 smem_tiled_copy_Q,
                            STensor1& sQ_copypartition, RTensor1& rQ_copy_view,
                            SmemTiledCopy2 smem_tiled_copy_K,
                            STensor2& sK_copypartition, RTensor2& rK_copy_view,
                            TiledMMAType tiled_mma1, RTensor3& rQ, RTensor4& rK,
                            RTensor5& acc_s_fragment, int sQ_stride,
                            int sK_stride, int num_stage = 2)
        : smem_tiled_copy_Q(smem_tiled_copy_Q),
          sQ_copypartition(sQ_copypartition),
          rQ_copy_view(rQ_copy_view),
          smem_tiled_copy_K(smem_tiled_copy_K),
          sK_copypartition(sK_copypartition),
          rK_copy_view(rK_copy_view),
          tiled_mma1(tiled_mma1),
          rQ(rQ),
          rK(rK),
          acc_s_fragment(acc_s_fragment),
          sQ_stride(sQ_stride),
          sK_stride(sK_stride),
          cur_iter(0),
          num_stage(num_stage) {}
    inline __device__ void prologue() {
        cur_iter = 0;
        cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, _0{}),
                   rQ_copy_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, _0{}),
                   rK_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(rK); ++i) {
            if (i < size<2>(rK) - 1) {
                cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, i + 1),
                           rQ_copy_view(_, _, i + 1));
                cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, i + 1),
                           rK_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma1, rQ(_, _, i), rK(_, _, i), acc_s_fragment);
        }
        sQ_copypartition.data() = sQ_copypartition.data() + sQ_stride;
        sK_copypartition.data() = sK_copypartition.data() + sK_stride;
        cur_iter++;
    }
    inline __device__ void body() {
        cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, _0{}),
                   rQ_copy_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, _0{}),
                   rK_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(rK); ++i) {
            if (i < size<2>(rK) - 1) {
                cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, i + 1),
                           rQ_copy_view(_, _, i + 1));
                cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, i + 1),
                           rK_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma1, rQ(_, _, i), rK(_, _, i), acc_s_fragment);
        }
        sQ_copypartition.data() = sQ_copypartition.data() + sQ_stride;
        sK_copypartition.data() = sK_copypartition.data() + sK_stride;
        if ((cur_iter + 1) % num_stage == 0) {
            sQ_copypartition.data() =
                sQ_copypartition.data() + (-sQ_stride * num_stage);
            sK_copypartition.data() =
                sK_copypartition.data() + (-sK_stride * num_stage);
        }
        cur_iter++;
    }
    inline __device__ void epilogue() {
        cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, _0{}),
                   rQ_copy_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, _0{}),
                   rK_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(rK); ++i) {
            if (i < size<2>(rK) - 1) {
                cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, i + 1),
                           rQ_copy_view(_, _, i + 1));
                cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, i + 1),
                           rK_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma1, rQ(_, _, i), rK(_, _, i), acc_s_fragment);
        }
        sQ_copypartition.data() = sQ_copypartition.data() + sQ_stride;
        sK_copypartition.data() = sK_copypartition.data() + sK_stride;
        if ((cur_iter + 1) % num_stage == 0) {
            sQ_copypartition.data() =
                sQ_copypartition.data() + (-sQ_stride * num_stage);
            sK_copypartition.data() =
                sK_copypartition.data() + (-sK_stride * num_stage);
        }
        cur_iter++;
    }

   private:
    int cur_iter;
    SmemTiledCopy1 smem_tiled_copy_Q;
    STensor1& sQ_copypartition;
    RTensor1& rQ_copy_view;
    SmemTiledCopy2 smem_tiled_copy_K;
    STensor2& sK_copypartition;
    RTensor2& rK_copy_view;
    TiledMMAType tiled_mma1;
    RTensor3& rQ;
    RTensor4& rK;
    RTensor5& acc_s_fragment;
    int sQ_stride, sK_stride;
    int num_stage;
};

template <class SmemTiledCopy1, class STensor1, class RTensor1,
          class SmemTiledCopy2, class STensor2, class RTensor2,
          class TiledMMAType, class RTensor3, class RTensor4, class RTensor5>
class CopyTilesS2R {
   public:
    __device__ CopyTilesS2R(SmemTiledCopy1 smem_tiled_copy_Q,
                            STensor1& sQ_copypartition, RTensor1& rQ_copy_view,
                            SmemTiledCopy2 smem_tiled_copy_K,
                            STensor2& sK_copypartition, RTensor2& rK_copy_view,
                            TiledMMAType tiled_mma1, RTensor3& rQ, RTensor4& rK,
                            RTensor5& acc_s_fragment, int sQ_stride,
                            int sK_stride, int num_stage = 2)
        : smem_tiled_copy_Q(smem_tiled_copy_Q),
          sQ_copypartition(sQ_copypartition),
          rQ_copy_view(rQ_copy_view),
          smem_tiled_copy_K(smem_tiled_copy_K),
          sK_copypartition(sK_copypartition),
          rK_copy_view(rK_copy_view),
          tiled_mma1(tiled_mma1),
          rQ(rQ),
          rK(rK),
          acc_s_fragment(acc_s_fragment),
          sQ_stride(sQ_stride),
          sK_stride(sK_stride),
          cur_iter(0),
          num_stage(num_stage),
          cur_iter_sq(0) {}
    inline __device__ void prologue() {
        cur_iter = 0;
        cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, _0{}),
                   rQ_copy_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, _0{}),
                   rK_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(rK); ++i) {
            if (i < size<2>(rK) - 1) {
                cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, i + 1),
                           rQ_copy_view(_, _, i + 1));
                cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, i + 1),
                           rK_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma1, rQ(_, _, i), rK(_, _, i), acc_s_fragment);
        }
        sQ_copypartition.data() = sQ_copypartition.data() + sQ_stride;
        sK_copypartition.data() = sK_copypartition.data() + sK_stride;
        cur_iter++;
    }
    inline __device__ void body() {
        cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, _0{}),
                   rQ_copy_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, _0{}),
                   rK_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(rK); ++i) {
            if (i < size<2>(rK) - 1) {
                cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, i + 1),
                           rQ_copy_view(_, _, i + 1));
                cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, i + 1),
                           rK_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma1, rQ(_, _, i), rK(_, _, i), acc_s_fragment);
        }
        sQ_copypartition.data() = sQ_copypartition.data() + sQ_stride;
        sK_copypartition.data() = sK_copypartition.data() + sK_stride;
        if ((cur_iter + 1) % num_stage == 0) {
            sK_copypartition.data() =
                sK_copypartition.data() + (-sK_stride * num_stage);
        }
        cur_iter++;
        cur_iter_sq++;
    }
    inline __device__ void epilogue() {
        cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, _0{}),
                   rQ_copy_view(_, _, _0{}));
        cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, _0{}),
                   rK_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(rK); ++i) {
            if (i < size<2>(rK) - 1) {
                cute::copy(smem_tiled_copy_Q, sQ_copypartition(_, _, i + 1),
                           rQ_copy_view(_, _, i + 1));
                cute::copy(smem_tiled_copy_K, sK_copypartition(_, _, i + 1),
                           rK_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma1, rQ(_, _, i), rK(_, _, i), acc_s_fragment);
        }
        sQ_copypartition.data() =
            sQ_copypartition.data() + (-sQ_stride * cur_iter_sq);
        sK_copypartition.data() = sK_copypartition.data() + sK_stride;
        if ((cur_iter + 1) % num_stage == 0) {
            sK_copypartition.data() =
                sK_copypartition.data() + (-sK_stride * num_stage);
        }
        cur_iter++;
        cur_iter_sq = 0;
    }

   private:
    int cur_iter, cur_iter_sq;
    SmemTiledCopy1 smem_tiled_copy_Q;
    STensor1& sQ_copypartition;
    RTensor1& rQ_copy_view;
    SmemTiledCopy2 smem_tiled_copy_K;
    STensor2& sK_copypartition;
    RTensor2& rK_copy_view;
    TiledMMAType tiled_mma1;
    RTensor3& rQ;
    RTensor4& rK;
    RTensor5& acc_s_fragment;
    int sQ_stride, sK_stride;
    int num_stage;
};

template <class SmemTiledCopy2, class STensor2, class RTensor2,
          class TiledMMAType, class RTensor4, class RTensor5>
class CopyTileS2R {
   public:
    __device__ CopyTileS2R(SmemTiledCopy2 smem_tiled_copy_V,
                           STensor2& sV_copypartition, RTensor2& rV_copy_view,
                           TiledMMAType tiled_mma, RTensor4& rV,
                           RTensor5& acc_o_fragment, int sV_stride,
                           int num_stage = 2)
        : smem_tiled_copy_V(smem_tiled_copy_V),
          sV_copypartition(sV_copypartition),
          rV_copy_view(rV_copy_view),
          tiled_mma(tiled_mma),
          rV(rV),
          acc_o_fragment(acc_o_fragment),
          sV_stride(sV_stride),
          cur_iter(0),
          cur_iter_sv(0),
          num_stage(num_stage) {}
    template <class RTensor3>
    inline __device__ void prologue(RTensor3& rP) {
        cur_iter = 0;
        cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, _0{}),
                   rV_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(rV); ++i) {
            if (i < size<2>(rV) - 1) {
                cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, i + 1),
                           rV_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, rP(_, _, cur_iter * size<2>(rV) + i),
                       rV(_, _, i), acc_o_fragment);
        }
        sV_copypartition.data() = sV_copypartition.data() + sV_stride;
        cur_iter++;
    }
    template <class RTensor3>
    inline __device__ void body(RTensor3& rP) {
        cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, _0{}),
                   rV_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(rV); ++i) {
            if (i < size<2>(rV) - 1) {
                cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, i + 1),
                           rV_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, rP(_, _, cur_iter_sv * size<2>(rV) + i),
                       rV(_, _, i), acc_o_fragment);
        }
        sV_copypartition.data() = sV_copypartition.data() + sV_stride;
        if ((cur_iter + 1) % num_stage == 0) {
            sV_copypartition.data() =
                sV_copypartition.data() + (-sV_stride * num_stage);
        }
        cur_iter++;
        cur_iter_sv++;
    }
    template <class RTensor3>
    inline __device__ void epilogue(RTensor3& rP) {
        cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, _0{}),
                   rV_copy_view(_, _, _0{}));
#pragma unroll
        for (int i = 0; i < size<2>(rV); ++i) {
            if (i < size<2>(rV) - 1) {
                cute::copy(smem_tiled_copy_V, sV_copypartition(_, _, i + 1),
                           rV_copy_view(_, _, i + 1));
            }
            cute::gemm(tiled_mma, rP(_, _, cur_iter_sv * size<2>(rV) + i),
                       rV(_, _, i), acc_o_fragment);
        }
        sV_copypartition.data() = sV_copypartition.data() + sV_stride;
        if ((cur_iter + 1) % num_stage == 0) {
            sV_copypartition.data() =
                sV_copypartition.data() + (-sV_stride * num_stage);
        }
        cur_iter++;
        cur_iter_sv = 0;
    }

   private:
    int cur_iter, cur_iter_sv;
    SmemTiledCopy2 smem_tiled_copy_V;
    STensor2& sV_copypartition;
    RTensor2& rV_copy_view;
    TiledMMAType tiled_mma;
    RTensor4& rV;
    RTensor5& acc_o_fragment;
    int sV_stride;
    int num_stage;
};
