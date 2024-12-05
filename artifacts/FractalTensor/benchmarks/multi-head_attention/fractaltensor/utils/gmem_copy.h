#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

using namespace cute;

template <int N>
CUTE_HOST_DEVICE void cp_async_wait_flash() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

template <class GmemTiledCopy, class GTensor1, class RTensor1>
__device__ void copy_Global2Reg(GmemTiledCopy gmem_tiled_copy_QKV,
                                GTensor1& gQ_partition,
                                RTensor1& rQ_partition) {
    CUTE_STATIC_ASSERT_V(size<0>(gQ_partition) == size<0>(rQ_partition));
    CUTE_STATIC_ASSERT_V(size<1>(gQ_partition) == size<2>(rQ_partition));
#pragma unroll
    for (int i = 0; i < size<1>(gQ_partition); i++) {
        cute::copy(gmem_tiled_copy_QKV, gQ_partition(_, i, _0{}),
                   rQ_partition(_, _0{}, i));
    }
}

template <class GmemTiledCopy, class GTensor1, class RTensor1>
__device__ void copy_Reg2Global(GmemTiledCopy gmem_tiled_copy_QKV,
                                RTensor1& rQ_partition,
                                GTensor1& gQ_partition) {
    CUTE_STATIC_ASSERT_V(size<0>(gQ_partition) == size<0>(rQ_partition));
    CUTE_STATIC_ASSERT_V(size<1>(gQ_partition) == size<2>(rQ_partition));
#pragma unroll
    for (int i = 0; i < size<1>(gQ_partition); i++) {
        cute::copy(gmem_tiled_copy_QKV, rQ_partition(_, _0{}, i),
                   gQ_partition(_, i, _0{}));
    }
}
