#pragma once

#include "kaleido/core/device/kernels/tiled_copy.h"

template <typename InType, const int kThreads, const int kDimK, const int kDimV,
          const int kTileSizeRow, const int kTileSizeCol,
          const int BlockKSmem = kDimK, const int num_stages_qk = 1,
          const bool load_q_once = true, const int BlockKSmem2 = kTileSizeCol,
          const int num_stages_v = 1, const int SmemKAtom = 64,
          const int kSwizzle = 3, const bool unrollLastIter = false>
struct KeTraits {
    static_assert(kDimK % (BlockKSmem) == 0, "kDimK%(BlockKSmem)!=0");
    static_assert(kTileSizeCol % (BlockKSmem2) == 0,
                  "kTileSizeCol%(BlockKSmem2)!=0");

    static_assert(BlockKSmem % SmemKAtom == 0, "BlockKSmem%SmemKAtom!=0");
    static_assert(BlockKSmem2 % (kThreads / (SmemKAtom / 8)) == 0,
                  "gmem load V fail");
    static_assert(kTileSizeRow % (kThreads / (SmemKAtom / 8)) == 0,
                  "gmem load Q fail");
    static_assert(kTileSizeCol % (kThreads / (SmemKAtom / 8)) == 0,
                  "gmem load K fail");

    constexpr static int kWarps = kThreads / 32;

    using TiledMma =
        TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
                 Layout<Shape<Int<kWarps>, _1, _1>>, Layout<Shape<_1, _2, _1>>>;

    using GmemCopyLayoutAtom =
        Layout<Shape<Int<kThreads / (SmemKAtom / 8)>, Int<SmemKAtom / 8>>,
               Stride<Int<SmemKAtom / 8>, _1>>;
    using CopyG2S = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, InType>{},
        GmemCopyLayoutAtom{}, Layout<Shape<_1, _8>>{}));

    using SmemLayoutAtom = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<_8, Int<SmemKAtom>>, Stride<Int<SmemKAtom>, _1>>{}));

    using GmemLayoutQ = Layout<Shape<Int<kTileSizeRow>, Int<BlockKSmem>>,
                               Stride<Int<kDimK>, _1>>;

    using GmemLayoutK = Layout<Shape<Int<kTileSizeCol>, Int<BlockKSmem>>,
                               Stride<Int<kDimK>, _1>>;

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kTileSizeRow>, Int<BlockKSmem>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kTileSizeCol>, Int<BlockKSmem>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, InType>;

    using SmemLayoutAtomVtransposedNoSwizzle =
        Layout<Shape<Int<SmemKAtom>, Int<BlockKSmem2>>,
               Stride<_1, Int<SmemKAtom>>>;

    using SmemLayoutAtomVtransposed = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{}, SmemLayoutAtomVtransposedNoSwizzle{}));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<BlockKSmem2>, Int<kDimV>>{}));

    using SmemLayoutVtransposed = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed{}, Shape<Int<kDimV>, Int<BlockKSmem2>>{}));

    using SmemLayoutVtransposedNoSwizzle =
        decltype(tile_to_shape(SmemLayoutAtomVtransposedNoSwizzle{},
                               Shape<Int<kDimV>, Int<BlockKSmem2>>{}));

    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, InType>;

    using SmemLayoutAtomO = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<Int<8>, Int<SmemKAtom>>, Stride<Int<SmemKAtom>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{}, Shape<Int<kTileSizeRow>, Int<kDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, InType>;

    using GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, InType>{}, GmemCopyLayoutAtom{},
        Layout<Shape<_1, _8>>{}));
};
