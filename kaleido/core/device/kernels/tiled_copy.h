#pragma once

#include "kaleido/core/device/kernels/tiled_copy.h"
#include "kaleido/core/device/traits_base.h"
#include "kaleido/core/layout.h"
#include "kaleido/core/tile_shape.h"

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

namespace kaleido {
namespace core {
namespace cuda_kernel {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_ENABLED
#endif

template <int N>
DEVICE void wait_group() {
#if defined(CP_ASYNC_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

DEVICE
void commit_copy_group() {
#if defined(CP_ASYNC_ENABLED)
    cute::cp_async_fence();
#endif
}

DEVICE
void __copy_async() {
    commit_copy_group();
    wait_group<0>();
}

// TODO(ying): Figure out how to use the same interface for both G2S
// and S2G
// @param ThreadsShape: the shape of the thread block, it should has
// a type of TileShape.
template <typename Element, typename ThreadsShape, typename SrcLayout,
          typename DstLayout, typename Base = TraitsBase<Element>>
struct G2SCopy2D : public Base {
    using SrcLayout_ = SrcLayout;
    using DstLayout_ = DstLayout;

    static constexpr int kThreadsPerRow = dim_size<0, ThreadsShape>;
    static constexpr int kThreadsPerCol = dim_size<1, ThreadsShape>;

    using ThreadLayout = Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
                                Stride<Int<kThreadsPerCol>, _1>>;
#if defined(CP_ASYNC_ENABLED)
    using CopyInst =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
#else
    using CopyInst = Copy_Atom<DefaultCopy, Element>;
#endif
    using TiledCopy = decltype(make_tiled_copy(
        CopyInst{}, ThreadLayout{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

   public:
    DEVICE void copy(const Element* src_data, Element* trg_data, int tid) {
        auto gtile = make_tensor(make_gmem_ptr(src_data), SrcLayout_{});
        auto stile = make_tensor(make_smem_ptr(trg_data), DstLayout_{});

        auto loader = tiled_copy_.get_thread_slice(tid);

        auto src = loader.partition_S(gtile);
        auto dst = loader.partition_D(stile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
            for (int j = 0; j < int(size<2>(src)); ++j)
                cute::copy(tiled_copy_, src(_, i, j), dst(_, i, j));
    }

   private:
    TiledCopy tiled_copy_;
};

// @param ThreadsShape: the shape of the thread block, it should has
// a type of TileShape.
template <typename Element, typename ThreadsShape, typename SrcLayout,
          typename DstLayout, typename Base = TraitsBase<Element>>
struct S2GCopy2D : public Base {
    using SrcLayout_ = SrcLayout;
    using DstLayout_ = DstLayout;

    static constexpr int kThreadsPerRow = dim_size<0, ThreadsShape>;
    static constexpr int kThreadsPerCol = dim_size<1, ThreadsShape>;

    using ThreadLayout = Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
                                Stride<Int<kThreadsPerCol>, _1>>;

    using TiledCopy = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{}, ThreadLayout{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

   public:
    DEVICE void copy(const Element* src_data, Element* trg_data, int tid) {
        auto stile = make_tensor(make_smem_ptr(src_data), SrcLayout_{});
        auto gtile = make_tensor(make_gmem_ptr(trg_data), DstLayout_{});

        auto loader = tiled_copy_.get_thread_slice(tid);

        auto src = loader.partition_S(stile);
        auto dst = loader.partition_D(gtile);

#pragma unroll
        for (int i = 0; i < int(size<1>(src)); ++i)
#pragma unroll
            for (int j = 0; j < int(size<2>(src)); ++j)
                cute::copy(tiled_copy_, src(_, i, j), dst(_, i, j));
    }

   private:
    TiledCopy tiled_copy_;
};

template <typename TiledCopy, typename STensor, typename DTensor,
          typename DTensorView>
struct Shm2RegLoad {
   public:
    DEVICE
    Shm2RegLoad(TiledCopy& copy, const STensor& src, DTensor& dst,
                DTensorView& dst_view)
        : tiled_copy_(copy), src_(src), dst_(dst), dst_view_(dst_view) {}

    DEVICE void copy(int pos) {
        cute::copy(tiled_copy_, src_(_, _, pos), dst_view_(_, _, pos));
    }

    DEVICE int get_iters() { return size<2>(dst_); }

    DEVICE const auto& operator[](int idx) { return dst_(_, _, idx); }

   private:
    TiledCopy& tiled_copy_;
    const STensor& src_;
    DTensor& dst_;
    DTensorView& dst_view_;
};

template <typename Element, typename Layout, typename TiledMma>
DEVICE auto& make_s2rA(const Element* data, int tid, const Layout& layout,
                       const TiledMma& tiled_mma) {
    auto tensor = make_tensor(make_smem_ptr(data), layout);

    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto tiled_copy = make_tiled_copy_A(SmemLoadAtom{}, tiled_mma);

    auto thrd_copy = tiled_copy.get_thread_slice(tid);
    auto src = thrd_copy.partition_S(tensor);

    // partition register
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto dst = thr_mma.partition_fragment_A(tensor);
    auto dst_view = thrd_copy.retile_S(dst);

    Shm2RegLoad loader(tiled_copy, src, dst, dst_view);
    return loader;
}

/// FIXIME(ying): the current implementation is for fast experiment, it is
/// coupled shared memory layout with the register layout
template <typename Element, typename Layout, typename TiledMma>
DEVICE auto& make_s2rB(const Element* data, int tid, const Layout& layout,
                       const TiledMma& tiled_mma) {
    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    auto tiled_copy = make_tiled_copy_B(SmemLoadAtom{}, tiled_mma);
    auto thrd_copy = tiled_copy.get_thread_slice(tid);

    auto tensor = make_tensor(make_smem_ptr(data), layout);
    auto src = thrd_copy.partition_S(tensor);

    // partition register
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto dst = thr_mma.partition_fragment_B(tensor);
    auto dst_view = thrd_copy.retile_D(dst);

    Shm2RegLoad loader(tiled_copy, src, dst, dst_view);
    return loader;
}

template <typename Element, typename TiledMma_, typename DstLayout>
struct R2SCopy2D {
    // the shared layout is determined by the tiled mma
    using TiledMma = TiledMma_;
    using DstLayout_ = DstLayout;
    using CopyAtom = Copy_Atom<DefaultCopy, Element>;

   public:
    template <typename Engine, typename Layout>
    DEVICE void copy(cute::Tensor<Engine, Layout> const& acc, Element* dst_data,
                     int tid) {
        // FIXME(ying): This implementation is specifically designed
        // for TCU WMMA and assumes that the ACC value has a
        // floating-point precision. The code converts the ACC value
        // to half-precision.
        auto src_tensor = convert_type<Element>(acc);
        auto dst_tensor = make_tensor(make_smem_ptr(dst_data), DstLayout{});

        auto tiled_copy = make_tiled_copy_C(CopyAtom{}, TiledMma{});
        auto thrd_copy = tiled_copy.get_thread_slice(tid);

        auto src = thrd_copy.retile_S(src_tensor);
        auto dst = thrd_copy.partition_D(dst_tensor);
        cute::copy(tiled_copy, src, dst);
    }

   private:
    template <typename To_type, typename Engine, typename Layout>
    CUTE_DEVICE auto convert_type(cute::Tensor<Engine, Layout> const& tensor) {
        using From_type = typename Engine::value_type;
        constexpr int numel = decltype(size(tensor))::value;
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        // HACK: this requires tensor to be "contiguous"
        auto frag = convert_op(
            *reinterpret_cast<const cutlass::Array<From_type, numel>*>(
                tensor.data()));
        return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    }
};

template <class GmemTiledCopy, class GTensor1, class STensor1, class GTensor2,
          class STensor2>
class CopyTilesG2S {
   public:
    __device__ CopyTilesG2S(GmemTiledCopy gmem_tiled_copy_QKV,
                            GTensor1& gQ_partition, STensor1& sQ_partition,
                            GTensor2& gK_partition, STensor2& sK_partition,
                            int gQ_stride, int sQ_stride, int gK_stride,
                            int sK_stride, int num_stage = 2)
        : gmem_tiled_copy_QKV(gmem_tiled_copy_QKV),
          gQ_partition(gQ_partition),
          sQ_partition(sQ_partition),
          gK_partition(gK_partition),
          sK_partition(sK_partition),
          gQ_stride(gQ_stride),
          sQ_stride(sQ_stride),
          gK_stride(gK_stride),
          sK_stride(sK_stride),
          cur_iter(0),
          num_stage(num_stage) {}

    inline __device__ void prologue() {
#pragma unroll
        for (int m = 0; m < size<1>(gQ_partition); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gQ_partition); ++k) {
                cute::copy(gmem_tiled_copy_QKV, gQ_partition(_, m, k),
                           sQ_partition(_, m, k));
            }
        }
#pragma unroll
        for (int m = 0; m < size<1>(gK_partition); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gK_partition); ++k) {
                cute::copy(gmem_tiled_copy_QKV, gK_partition(_, m, k),
                           sK_partition(_, m, k));
            }
        }
        cute::cp_async_fence();
        gQ_partition.data() = gQ_partition.data() + gQ_stride;
        gK_partition.data() = gK_partition.data() + gK_stride;
        sQ_partition.data() = sQ_partition.data() + sQ_stride;
        sK_partition.data() = sK_partition.data() + sK_stride;
        if ((cur_iter + 1) % num_stage == 0) {
            sK_partition.data() =
                sK_partition.data() + (-sK_stride * num_stage);
            sQ_partition.data() =
                sQ_partition.data() + (-sQ_stride * num_stage);
        }
        cur_iter++;
    }

    inline __device__ void body() {
#pragma unroll
        for (int m = 0; m < size<1>(gQ_partition); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gQ_partition); ++k) {
                cute::copy(gmem_tiled_copy_QKV, gQ_partition(_, m, k),
                           sQ_partition(_, m, k));
            }
        }
#pragma unroll
        for (int m = 0; m < size<1>(gK_partition); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gK_partition); ++k) {
                cute::copy(gmem_tiled_copy_QKV, gK_partition(_, m, k),
                           sK_partition(_, m, k));
            }
        }
        cute::cp_async_fence();
        gQ_partition.data() = gQ_partition.data() + gQ_stride;
        gK_partition.data() = gK_partition.data() + gK_stride;
        sK_partition.data() = sK_partition.data() + (sK_stride);
        sQ_partition.data() = sQ_partition.data() + (sQ_stride);
        if ((cur_iter + 1) % num_stage == 0) {
            sK_partition.data() =
                sK_partition.data() + (-sK_stride * num_stage);
            sQ_partition.data() =
                sQ_partition.data() + (-sQ_stride * num_stage);
        }
        cur_iter++;
    }

    inline __device__ void epilogue() {
#pragma unroll
        for (int m = 0; m < size<1>(gQ_partition); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gQ_partition); ++k) {
                cute::copy(gmem_tiled_copy_QKV, gQ_partition(_, m, k),
                           sQ_partition(_, m, k));
            }
        }
#pragma unroll
        for (int m = 0; m < size<1>(gK_partition); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gK_partition); ++k) {
                cute::copy(gmem_tiled_copy_QKV, gK_partition(_, m, k),
                           sK_partition(_, m, k));
            }
        }
        cute::cp_async_fence();
    }

   private:
    int cur_iter;
    GmemTiledCopy gmem_tiled_copy_QKV;
    GTensor1& gQ_partition;
    STensor1& sQ_partition;
    GTensor2& gK_partition;
    STensor2& sK_partition;
    int gQ_stride, sQ_stride, gK_stride, sK_stride;
    int num_stage;
};

template <class GmemTiledCopy, class GTensor1, class STensor1>
class CopyTileG2S {
   public:
    __device__ CopyTileG2S(GmemTiledCopy gmem_tiled_copy_QKV,
                           GTensor1& gV_partition, STensor1& sV_partition,
                           int gV_stride, int sV_stride, int num_stage = 2)
        : gmem_tiled_copy_QKV(gmem_tiled_copy_QKV),
          gV_partition(gV_partition),
          sV_partition(sV_partition),
          gV_stride(gV_stride),
          sV_stride(sV_stride),
          cur_iter(0),
          num_stage(num_stage) {}

    inline __device__ void prologue() {
#pragma unroll
        for (int m = 0; m < size<1>(gV_partition); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gV_partition); ++k) {
                cute::copy(gmem_tiled_copy_QKV, gV_partition(_, m, k),
                           sV_partition(_, m, k));
            }
        }
        cute::cp_async_fence();
        gV_partition.data() = gV_partition.data() + gV_stride;
        sV_partition.data() = sV_partition.data() + sV_stride;
        if ((cur_iter + 1) % num_stage == 0) {
            sV_partition.data() =
                sV_partition.data() + (-sV_stride * num_stage);
        }
        cur_iter++;
    }

    inline __device__ void body() {
#pragma unroll
        for (int m = 0; m < size<1>(gV_partition); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gV_partition); ++k) {
                cute::copy(gmem_tiled_copy_QKV, gV_partition(_, m, k),
                           sV_partition(_, m, k));
            }
        }
        cute::cp_async_fence();
        gV_partition.data() = gV_partition.data() + gV_stride;
        sV_partition.data() = sV_partition.data() + sV_stride;

        if ((cur_iter + 1) % num_stage == 0) {
            sV_partition.data() =
                sV_partition.data() + (-sV_stride * num_stage);
        }
        cur_iter++;
    }

    inline __device__ void epilogue() {
#pragma unroll
        for (int m = 0; m < size<1>(gV_partition); ++m) {
#pragma unroll
            for (int k = 0; k < size<2>(gV_partition); ++k) {
                cute::copy(gmem_tiled_copy_QKV, gV_partition(_, m, k),
                           sV_partition(_, m, k));
            }
        }
        cute::cp_async_fence();
    }

   private:
    int cur_iter;
    GmemTiledCopy gmem_tiled_copy_QKV;
    GTensor1& gV_partition;
    STensor1& sV_partition;
    int gV_stride, sV_stride;
    int num_stage;
};

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
