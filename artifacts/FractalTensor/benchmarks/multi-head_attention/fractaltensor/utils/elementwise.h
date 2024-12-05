#pragma once
#include "reduce.h"

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>

using namespace cute;

template <class SmemTiledCopy1, class STensor1, class RTensor1, class Tensor0,
          class Tensor1>
__device__ inline void multiply_mask(SmemTiledCopy1 smem_tiled_copy_mask,
                                     STensor1& sMask_copypartition,
                                     RTensor1& rMask_copy_view,
                                     Tensor0& acc_s_fragment, Tensor1& rMask) {
    using namespace cute;
    cute::copy(smem_tiled_copy_mask, sMask_copypartition(_, _, _0{}),
               rMask_copy_view(_, _, _0{}));
#pragma unroll
    for (int ax0 = 0; ax0 < size<2>(acc_s_fragment); ax0++) {
        if (ax0 < size<2>(acc_s_fragment) - 1) {
            cute::copy(smem_tiled_copy_mask, sMask_copypartition(_, _, ax0 + 1),
                       rMask_copy_view(_, _, ax0 + 1));
        }
#pragma unroll
        for (int ax1 = 0; ax1 < size<1>(acc_s_fragment); ax1++) {
#pragma unroll
            for (int ax2 = 0; ax2 < size<0>(acc_s_fragment); ax2++) {
                acc_s_fragment(ax2, ax1, ax0) =
                    acc_s_fragment(ax2, ax1, ax0) *
                    __half2float(rMask(ax2, ax1, ax0));
            }
        }
    }
}

template <int RThreads, class Tensor0, class Tensor1, class Tensor2>
__device__ inline void update_r(Tensor0& r_new_fragment,
                                Tensor1& r_wo_clamp_fragment, Tensor2& scores) {
    using namespace cute;
    Tensor r_wo_clamp_fragment_tmp = make_fragment_like(r_wo_clamp_fragment);
    reduce_sumabs<RThreads>(scores, r_wo_clamp_fragment_tmp);
#pragma unroll
    for (int ax0 = 0; ax0 < size<0>(r_wo_clamp_fragment); ax0++) {
        r_wo_clamp_fragment(ax0) += r_wo_clamp_fragment_tmp(ax0);
    }

#pragma unroll
    for (int ax0 = 0; ax0 < size<0>(r_wo_clamp_fragment); ax0++) {
        r_new_fragment(ax0) = max(r_wo_clamp_fragment(ax0), 1.0f);
    }
}
