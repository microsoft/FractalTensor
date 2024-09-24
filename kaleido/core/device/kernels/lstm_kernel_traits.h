#pragma once
#include "kaleido/core/device/traits_base.h"

namespace kaleido {
namespace core {
namespace cuda_kernel {

using namespace cute;

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape,
          typename Base = TraitsBase<Element_>>
struct GemmAddTraits : public Base {
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

    static constexpr int kSwizzle = (kTK == 32 ? 2 : 3);
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<kTK>>, Stride<Int<kTK>, _1>>{}));
    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutD =
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

    using LoadC_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTM, kTK, kK>, SmemLayoutA>;

    using LoadD_G2S =
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

    using StoreE_R2S = R2SCopy2D<Element, TiledMma, RowMajor<kTM, kTN>>;

    using StoreE_S2G =
        S2GCopy2D<Element, ThreadShape,
                  decltype(tile_to_shape(
                      SmemLayoutAtom{},
                      Shape<Int<kTN>, Int<kTK>>{})) /*shared memory layout*/,
                  RowMajor<kTM, kTN, kN> /*global memory layout*/>;
};

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape,
          typename Base = TraitsBase<Element_>>
using GemmAddSigmodTraits =
    GemmAddTraits<Element_, InstructionShape, ValueMnk, WarpArrangement,
                  CtaTileShape, WholeShape, Base>;

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape,
          typename Base = TraitsBase<Element_>>
using GemmAddTanhTraits =
    GemmAddTraits<Element_, InstructionShape, ValueMnk, WarpArrangement,
                  CtaTileShape, WholeShape, Base>;

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape,
          typename Base = TraitsBase<Element_>>
using LSTMLayerTraits =
    GemmAddTraits<Element_, InstructionShape, ValueMnk, WarpArrangement,
                  CtaTileShape, WholeShape, Base>;

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape,
          typename Base = TraitsBase<Element_>>
struct CuteLSTMGemmTraits : public Base {
    using Element = Element_;
    // kM: 4 * hidden_size
    static constexpr int kM = dim_size<0, WholeShape>;
    // kN: batch_size * seq_length
    static constexpr int kN = dim_size<1, WholeShape>;
    // kK: hidden_size
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
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutD =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));

    using ThreadShape = TileShape<kThreadsPerRow, kThreadsPerCol>;
    using LoadW_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTM, kTK, kK>, SmemLayoutA>;

    // NOTE: the input matrix B: [kK, kN] is physically laid out in
    // a column major format, that is, the K dimension is contiguous
    // in memory. However, a physically column major matrix can be
    // viewed as a row major matrix with a transpose. Therefore, we
    // can use the `RowMajor` layout here.
    //   using ThreadShapeB = TileShape<kThreadsPerRow,
    //   kThreadsPerCol>;
    using LoadX_G2S =
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

    using StoreO_R2S = R2SCopy2D<Element, TiledMma, RowMajor<kTM, kTN>>;

    using StoreO_S2G =
        S2GCopy2D<Element, ThreadShape,
                  RowMajor<kTM, kTN> /*shared memory layout*/,
                  RowMajor<kTM, kTN, kN> /*global memory layout*/>;
};

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape, typename WholeShape,
          typename Base = TraitsBase<Element_>>
struct BatchedLSTMGateTraits : public Base {
    using Element = Element_;
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
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutD =
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

    using LoadC_G2S =
        G2SCopy2D<Element, ThreadShape, RowMajor<kTM, kTK, kK>, SmemLayoutA>;

    using LoadD_G2S =
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

    using SmemLayoutE =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTN>>{}));

    using StoreE_R2S = R2SCopy2D<Element, TiledMma, SmemLayoutD>;

    using StoreE_S2G =
        S2GCopy2D<Element, ThreadShape, SmemLayoutE /*shared memory
        layout*/, RowMajor<kTM, kTN, kN> /*global memory layout*/>;
};

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape,
          typename Base = TraitsBase<Element_>>
struct DynamicBatchedLSTMGateTraits : public Base {
    using Element = Element_;

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
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTK>>{}));
    using SmemLayoutD =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTN>, Int<kTK>>{}));

    using ThreadShape = TileShape<kThreadsPerRow, kThreadsPerCol>;

    using ThreadLayout = Layout<Shape<Int<kThreadsPerRow>, Int<kThreadsPerCol>>,
                                Stride<Int<kThreadsPerCol>, _1>>;

    static const bool enable_cp_async = false;  // change this flag
    using CopyInst = std::conditional_t<
        enable_cp_async,
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>,
        Copy_Atom<DefaultCopy, Element>>;

    using TiledCopy = decltype(make_tiled_copy(
        CopyInst{}, ThreadLayout{},
        Layout<Shape<_1, Int<Base::kNumPerAccess>>>{}));

    // TODO(ying): The current implementation uses ldmatrix.x4
    // instruction which requires the TileMMA configuration to be
    // fixed as follows. Make it able to be tuned by policy in
    // future implementation.
    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarpPerRow>, Int<kWarpPerCol>, _1>>,
                 Layout<Shape<_1, _2, _1>>>;
    using SmemLoadAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

    using SmemLayoutE =
        decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kTM>, Int<kTN>>{}));

    using StoreE_R2S = R2SCopy2D<Element, TiledMma, SmemLayoutD>;
};

template <typename Element_, typename InstructionShape, typename ValueMnk,
          typename WarpArrangement, typename CtaTileShape,
          typename Base = TraitsBase<Element_>>
using DynamicLstmGateTraits =
    DynamicBatchedLSTMGateTraits<Element_, InstructionShape, ValueMnk,
                                 WarpArrangement, CtaTileShape, Base>;

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
