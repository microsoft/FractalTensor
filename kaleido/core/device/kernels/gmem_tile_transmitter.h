#pragma once
#include "kaleido/core/device/kernels/tile_transmitter.h"

namespace kaleido {
namespace core {
namespace cuda_kernel {

using namespace cute;

// All threads in a CTA loads a data tile from GLOBAL MEMORY to
// SHARED MEMORY
template <const int kRow, const int kCol, typename Element_, const int kThreads,
          TileLayout kSrcLayout, TileLayout kTrgLayout>
class GmemTileTransmitter {
   public:
    using Element = Element_;

    GmemTileTransmitter(int64_t row_size, int64_t col_size)
        : row_size(row_size), col_size(col_size) {}

    __device__ void transfer(const Element* src, Element* trg, int stride,
                             int tid) const;

    int64_t row_size;
    int64_t col_size;
};

template <const int kRow, const int kCol, typename Element_, const int kThreads>
class GmemTileTransmitter<kRow, kCol, Element_, kThreads, TileLayout::RowMajor,
                          TileLayout::RowMajor> {
   public:
    using Element = Element_;

    static const int kAccessInBits = 128;
    static const int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static const int kNumPerAccess = kAccessInBits / kElmentBits;

    static const int kWarpContiguous = 4;  // 4 x 8, 8 x 4, or 8 x 8
    static const int kWarpStride = 32 / kWarpContiguous;

    using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
        cutlass::layout::PitchLinearShape<kCol, kRow>, kThreads,
        cutlass::layout::PitchLinearShape<kWarpContiguous, kWarpStride>,
        kNumPerAccess>;

    using GmemIterator =
        cutlass::transform::threadblock::PredicatedTileIterator<
            cutlass::MatrixShape<kRow, kCol>, Element,
            cutlass::layout::RowMajor, 1, ThreadMap>;

    using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<kRow, kCol>, Element, cutlass::layout::RowMajor, 1,
        ThreadMap>;

    using Fragment = typename GmemIterator::Fragment;

    GmemTileTransmitter(int64_t row_size, int64_t col_size)
        : row_size(row_size), col_size(col_size) {}

    __device__ void transfer(const Element* src, Element* trg, int stride,
                             int tid) const {
        typename GmemIterator::Params params({stride});
        GmemIterator gmem_iterator(params, const_cast<Element*>(src),
                                   {row_size, col_size}, tid);

        typename SmemIterator::TensorRef ref(
            trg, cutlass::layout::RowMajor(col_size));
        SmemIterator smem_iterator(ref, tid);

        Fragment frag;
        frag.clear();
        gmem_iterator.load(frag);
        smem_iterator.store(frag);
        __syncthreads();
    };

    int64_t row_size;
    int64_t col_size;
};

/// RowMajor to Swizzled RowMajor
template <const int kRow, const int kCol, typename Element_, const int kThreads>
class GmemTileTransmitter<kRow, kCol, Element_, kThreads, TileLayout::RowMajor,
                          TileLayout::SwizzledRowMajor> {
   public:
    using Element = Element_;

    static const int kAccessInBits = 128;
    static const int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static const int kNumPerAccess = kAccessInBits / kElmentBits;

    // warp arrangement: 4 x 8, 8 x 4, 8 x 8
    static const int kWarpContiguous = 4;
    static const int kWarpStride = 32 / kWarpContiguous;

    using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
        cutlass::layout::PitchLinearShape<kCol, kRow>, kThreads,
        cutlass::layout::PitchLinearShape<kWarpContiguous, kWarpStride>,
        kNumPerAccess>;

    using GTileShape = cutlass::MatrixShape<kRow, kCol>;
    using GLayout = cutlass::layout::RowMajor;
    using GmemIterator =
        cutlass::transform::threadblock::PredicatedTileIterator<
            GTileShape, Element, GLayout, 1 /*AdvanceRank*/, ThreadMap>;

    static const int SmemCacheLineWidth = 128;  // 128B
    // FIXME(ying): A hard-coded implementation that partitions a
    // cache line into 2 groups to do swizzling.
    static const int crosswise = SmemCacheLineWidth * 8 / kElmentBits / 2;

    using SLayout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<Element>::value, crosswise>;
    using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<kRow, kCol>, Element, SLayout, 1, ThreadMap>;

    using Fragment = typename GmemIterator::Fragment;

    GmemTileTransmitter(int64_t row_size, int64_t col_size)
        : row_size(row_size), col_size(col_size) {}

    __device__ void transfer(const Element* src, Element* trg, int stride,
                             int tid) const {
        typename GmemIterator::Params params({stride});
        GmemIterator gmem_iterator(params, const_cast<Element*>(src),
                                   {row_size, col_size}, tid);

        typename SmemIterator::TensorRef ref(
            trg, SLayout::packed({row_size, col_size}));
        SmemIterator smem_iterator(ref, tid);

        Fragment frag;
        frag.clear();
        gmem_iterator.load(frag);
        smem_iterator.store(frag);
        __syncthreads();
    };

    int64_t row_size;
    int64_t col_size;
};

#define FT(x) float(x * 1.0_hf)

/// RowMajor to SwizzledColumnMajor
template <const int kRow, const int kCol, typename Element_, const int kThreads>
class GmemTileTransmitter<kRow, kCol, Element_, kThreads, TileLayout::RowMajor,
                          TileLayout::SwizzledColumnMajor> {
   public:
    using Element = Element_;

    static const int kAccessInBits = 128;
    static const int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static const int kNumPerAccess = kAccessInBits / kElmentBits;

    static const int kWarpArrangeContiguous = 4;
    static const int kWarpArrangeStrided = 8;

    using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
        cutlass::layout::PitchLinearShape<kCol, kRow>, kThreads,
        cutlass::layout::PitchLinearShape<kWarpArrangeContiguous,
                                          kWarpArrangeStrided>,
        kNumPerAccess>;

    using GTileShape = cutlass::MatrixShape<kRow, kCol>;
    using GLayout = cutlass::layout::RowMajor;
    using GmemIterator =
        cutlass::transform::threadblock::PredicatedTileIterator<
            GTileShape, Element, GLayout, 1 /*AdvanceRank*/, ThreadMap>;

    static const int SmemCacheLineWidth = 128;  // 128B
    // FIXME(ying): A hard-coded implementation that partitions a
    // cache line into 2 groups to do swizzling.
    static const int crosswise = SmemCacheLineWidth * 8 / kElmentBits / 2;

    using SLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<Element>::value, crosswise>;
    using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<kCol, kRow>, Element, SLayout, 1, ThreadMap>;

    using Fragment = typename GmemIterator::Fragment;

    GmemTileTransmitter(int64_t row_size, int64_t col_size)
        : row_size(row_size), col_size(col_size) {}

    __device__ void transfer(const Element* src, Element* trg, int stride,
                             int tid) const {
        typename GmemIterator::Params params({stride});
        GmemIterator gmem_iterator(params, const_cast<Element*>(src),
                                   {row_size, col_size}, tid);
        SmemIterator smem_iterator(
            typename SmemIterator::TensorRef(
                {trg, SmemIterator::Layout::packed({col_size, row_size})}),
            tid);

        Fragment frag;
        frag.clear();
        gmem_iterator.load(frag);
        smem_iterator.store(frag);
        __syncthreads();
    };

    int64_t row_size;
    int64_t col_size;
};

/// ColumnMajor to SwizzledColumnMajor
template <const int kRow, const int kCol, typename Element_, const int kThreads>
class GmemTileTransmitter<kRow, kCol, Element_, kThreads,
                          TileLayout::ColumnMajor,
                          TileLayout::SwizzledColumnMajor> {
   public:
    using Element = Element_;

    static const int kAccessInBits = 128;
    static const int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static const int kNumPerAccess = kAccessInBits / kElmentBits;

    static const int kWarpArrangeContiguous = 4;
    static const int kWarpArrangeStrided = 8;

    // Define a global iterator, a shared iterator and their thread
    // map.
    using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
        cutlass::layout::PitchLinearShape<kRow, kCol> /*shape*/, kThreads,
        cutlass::layout::PitchLinearShape<
            kWarpArrangeContiguous, kWarpArrangeStrided> /*warp arrangement*/,
        kNumPerAccess>;

    using GTileShape = cutlass::MatrixShape<kRow, kCol>;
    using GLayout = cutlass::layout::ColumnMajor;
    using GmemIterator =
        cutlass::transform::threadblock::PredicatedTileIterator<
            GTileShape, Element, GLayout, 1 /*AdvanceRank*/, ThreadMap>;
    using Fragment = typename GmemIterator::Fragment;

    static const int SmemCacheLineWidth = 128;  // 128B
    // FIXME(ying): A hard-coded implementation that partitions a
    // cache line into 2 groups to do swizzling.
    static const int crosswise = SmemCacheLineWidth * 8 / kElmentBits / 2;

    using SLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<Element>::value, crosswise>;
    using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<kRow, kCol>, Element, SLayout, 0, ThreadMap>;

    // Ctor
    GmemTileTransmitter(int64_t row_size, int64_t col_size)
        : row_size(row_size), col_size(col_size) {}

    __device__ void transfer(const Element* src, Element* trg, int stride,
                             int tid) const {
        typename GmemIterator::Params params({stride});
        GmemIterator gmem_iterator(params, const_cast<Element*>(src),
                                   {row_size, col_size}, tid);
        SmemIterator smem_iterator(
            typename SmemIterator::TensorRef(
                {trg, SmemIterator::Layout::packed({row_size, col_size})}),
            tid);

        Fragment frag;
        frag.clear();
        gmem_iterator.load(frag);
        smem_iterator.store(frag);
        __syncthreads();
    };

    int64_t row_size;
    int64_t col_size;
};

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
