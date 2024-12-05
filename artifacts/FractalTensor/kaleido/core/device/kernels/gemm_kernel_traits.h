#pragma once

#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/tiled_copy.h"
#include "kaleido/core/device/traits_base.h"
#include "kaleido/core/layout.h"
#include "kaleido/core/tile_shape.h"

namespace kaleido {
namespace core {
namespace cuda_kernel {

using namespace cute;

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape,
          typename Base = TraitsBase<Element_>>
struct KeGemmTraits : public Base {
    using Element = Element_;
    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    static constexpr int kWarpPerRow = dim_size<0, WarpArrangement>;
    static constexpr int kWarpPerCol = dim_size<1, WarpArrangement>;
    static constexpr int kThreads = kWarpPerRow * kWarpPerCol * 32;

    static constexpr int kNumPerAccess = Base::kNumPerAccess;

    static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
    static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

    using SmemLayoutAtom = cute::Layout<Shape<_8, _32>, Stride<_32, _1>>;
    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));

    using ThreadShape = TileShape<kThreadsPerRow, kThreadsPerCol>;
    using LoadA_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTM, kTK, kK>, SmemLayoutA>;

    // NOTE: the input matrix B: [kK, kN] is physically laid out in
    // a column major format, that is, the K dimension is contiguous
    // in memory. However, a physically column major matrix can be
    // viewed as a row major matrix with a transpose. Therefore, we
    // can use the `RowMajor` layout here.
    //   using ThreadShapeB = TileShape<kThreadsPerRow,
    //   kThreadsPerCol>;
    using LoadB_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTN, kTK, kK>, SmemLayoutB>;

    // TODO(ying): The current implementation uses ldmatrix.x4
    // instruction which requires the TileMMA configuration to be
    // fixed as follows. Make it able to be tuned by policy in
    // future implementation.
    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Layout<Shape<_1, _2, _1>>>;
    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

    using StoreC_R2S = R2SCopy2D<Element, TiledMma, RowMajor<kTM, kTN>>;

    using StoreC_S2G =
        S2GCopy2D<Element, ThreadShape,
                  RowMajor<kTM, kTN> /*shared memory layout*/,
                  RowMajor<kTM, kTN, kN> /*global memory layout*/>;
};

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape,
          typename Base = TraitsBase<Element_>>
struct KeBatchedGemmTraits : public Base {
    using Element = Element_;
    // [M, N, K, B]
    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kB = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    static constexpr int kWarpPerRow = dim_size<0, WarpArrangement>;
    static constexpr int kWarpPerCol = dim_size<1, WarpArrangement>;
    static constexpr int kThreads = kWarpPerRow * kWarpPerCol * 32;

    static constexpr int kNumPerAccess = Base::kNumPerAccess;

    static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
    static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

    using SmemLayoutAtom = cute::Layout<Shape<_8, _32>, Stride<_32, _1>>;
    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));

    using ThreadShape = TileShape<kThreadsPerRow, kThreadsPerCol>;
    using LoadA_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTM, kTK, kK>, SmemLayoutA>;

    // NOTE: the input matrix B: [kK, kN] is physically laid out in
    // a column major format, that is, the K dimension is contiguous
    // in memory. However, a physically column major matrix can be
    // viewed as a row major matrix with a transpose. Therefore, we
    // can use the `RowMajor` layout here.
    //   using ThreadShapeB = TileShape<kThreadsPerRow,
    //   kThreadsPerCol>;
    using LoadB_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTN, kTK, kK>, SmemLayoutB>;

    // TODO(ying): The current implementation uses ldmatrix.x4
    // instruction which requires the TileMMA configuration to be
    // fixed as follows. Make it able to be tuned by policy in
    // future implementation.
    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Layout<Shape<_1, _2, _1>>>;
    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

    using StoreC_R2S = R2SCopy2D<Element, TiledMma, RowMajor<kTM, kTN>>;

    using StoreC_S2G =
        S2GCopy2D<Element, ThreadShape,
                  RowMajor<kTM, kTN> /*shared memory layout*/,
                  RowMajor<kTM, kTN, kN> /*global memory layout*/>;
};

template <typename Element_, const int num_stages_, const int kThreads,
          const int kWarpPerRow, typename CtaTileShape_, typename WholeShape_>
struct KePipelinedGemmTraits {
    using Element = Element_;
    using CtaTileShape = CtaTileShape_;
    using WholeShape = WholeShape_;

    static constexpr int num_stages = num_stages_;

    // the whole problem shape
    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;

    // the CTA tile shape
    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;

    // Declare global memory layout that is used to intepret the
    // input data in global memory.
    using GmemLayoutA = Layout<Shape<Int<kTM>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutB = Layout<Shape<Int<kTN>, Int<kTK>>, Stride<Int<kK>, _1>>;
    using GmemLayoutC = Layout<Shape<Int<kTM>, Int<kTN>>, Stride<Int<kN>, _1>>;

    static constexpr int kWarpSize = 32;
    static constexpr int kNumWarps = kThreads / kWarpSize;
    using TiledMma = TiledMMA<
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        Layout<Shape<Int<kNumWarps / kWarpPerRow>, Int<kWarpPerRow>, _1>>,
        Layout<Shape<_1, _2, _1>>>;
    // TODO(ying): using ldmatrix.x4 instruction is coupled with the
    // tensor core mma shape
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

    static constexpr int kAccessInBits = 128;
    static constexpr int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static constexpr int kNumPerAccess = kAccessInBits / kElmentBits;
    static constexpr int kThreadsPerRow = kTK / kNumPerAccess;
    static_assert(kThreadsPerRow <= kThreads,
                  "The tile size along the K dimension is too large for the "
                  "current implementation.");
    using GmemCopyLayoutAtom =
        Layout<Shape<Int<kThreads / kThreadsPerRow>, Int<kThreadsPerRow>>,
               Stride<Int<kThreadsPerRow>, _1>>;

    using GmemTiledCopy = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>{},
        GmemCopyLayoutAtom{}, Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    using GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{}, GmemCopyLayoutAtom{},
        Layout<Shape<_1, Int<kNumPerAccess>>>{}));

    // FIXME(ying). The swizzle function requires a data tile with a
    // minimal shape of <8, 32> for the <2, 3, 3> case, and a
    // minimal shape of <8, 64> for the <3, 3, 3> case. Here
    // requires some check to ensure that the data tile meets these
    // requirements before using this function.
    static constexpr int kSwizzle = (kTK == 32 ? 2 : 3);
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kTK>>, Stride<Int<kTK>, _1>>{}));
    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    using SmemLayoutO =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTN>>{}));

    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
};

template <typename Element_, typename WholeShape, typename CtaTileShape,
          typename WarpShape, typename Base = TraitsBase<Element_>>
struct KeBack2BackGemmTraits : public Base {
    using Element = Element_;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    static_assert(kTK == kTN && kTN == kTP,
                  "B2B GEMM requires kTK == kTN == kTP.");

    static constexpr int kWarpPerRow = dim_size<0, WarpShape>;
    static constexpr int kWarpPerCol = dim_size<1, WarpShape>;
    static_assert(kWarpPerCol == 1,
                  "The B2B GEMM requires a single warp along CTA tile.");

    static constexpr int kThreads = kWarpPerRow * kWarpPerCol * 32;

    static constexpr int kNumPerAccess = Base::kNumPerAccess;
    static constexpr int kThreadsPerCol = CeilDiv<kTK, Base::kNumPerAccess>;
    static constexpr int kThreadsPerRow = CeilDiv<kThreads, kThreadsPerCol>;

    static constexpr int kSwizzle = (kTK == 32 ? 2 : 3);
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kTK>>, Stride<Int<kTK>, _1>>{}));

    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));

    // The current implementation requires B are laid out in column
    // major. a [kTK, kTN] matrix in column major can be interpreted
    // as a [kTN, kTK] matrix in row major.
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    // a [kTN, kTP] matrix in column major fashion,
    // can be interpreted as a [kTP, kTN] matrix in row major
    // fashion.
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTP>, Int<kTN>>{}));

    using ThreadShape = TileShape<kThreadsPerRow, kThreadsPerCol>;
    using LoadA_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTM, kTK, kK>, SmemLayoutA>;
    using LoadB_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTN, kTK, kK>, SmemLayoutB>;
    using LoadC_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTP, kTN, kN>, SmemLayoutC>;

    // TODO(ying): The current implementation uses ldmatrix.x4
    // instruction which requires the TileMMA configuration to be
    // fixed as follows. Make it able to be tuned by policy in
    // future implementation.
    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Layout<Shape<_1, _2, _1>>>;

    using SmemLayoutD =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTP>>{}));
    using StoreD_R2S = R2SCopy2D<Element, TiledMma, SmemLayoutD>;
    using StoreD_S2G =
        S2GCopy2D<Element, ThreadShape, SmemLayoutD /*shared memory layout*/,
                  RowMajor<kTM, kTP, kP> /*global memory layout*/>;
};

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
