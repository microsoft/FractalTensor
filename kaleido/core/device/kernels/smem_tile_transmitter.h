#pragma once

#include "kaleido/core/device/kernels/tile_transmitter.h"

namespace kaleido {
namespace core {
namespace cuda_kernel {

/// All threads in a CTA read a SHARED memory tile that has a given
/// layout, and write to a GLOBAL memory tile that has a given
/// layout.
template <const int kRow, const int kCol, typename Element_, const int kThreads,
          TileLayout kSrcLayout, TileLayout kTrgLayout>
class SmemTileTransmitter {
   public:
    using Element = Element_;

    SmemTileTransmitter(int64_t row_size, int64_t col_size)
        : row_size(row_size), col_size(col_size) {}

    __device__ void transfer(const Element* src, Element* trg, int stride,
                             int tid) const;

    int64_t row_size;
    int64_t col_size;
};

template <const int kRow, const int kCol, typename Element_, const int kThreads>
class SmemTileTransmitter<kRow, kCol, Element_, kThreads, TileLayout::RowMajor,
                          TileLayout::RowMajor> {
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
            GTileShape, Element, GLayout, 1, ThreadMap>;

    using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<kRow, kCol>, Element, cutlass::layout::RowMajor, 1,
        ThreadMap>;

    using Fragment = typename GmemIterator::Fragment;

    SmemTileTransmitter(int64_t row_size, int64_t col_size)
        : row_size(row_size), col_size(col_size) {}

    __device__ void transfer(const Element* src, Element* trg, int stride,
                             int tid) const {
        typename SmemIterator::TensorRef ref(
            const_cast<Element*>(src), cutlass::layout::RowMajor(col_size));
        SmemIterator smem_iterator(ref, tid);

        typename GmemIterator::Params params({stride});
        GmemIterator gmem_iterator(params, trg, {row_size, col_size}, tid);

        Fragment frag;
        frag.clear();
        smem_iterator.load(frag);
        gmem_iterator.store(frag);
        __syncthreads();
    };

    int64_t row_size;
    int64_t col_size;
};

/// shared memory SwizzledRowMajor to global memory RowMajor
template <const int kRow, const int kCol, typename Element_, const int kThreads>
class SmemTileTransmitter<kRow, kCol, Element_, kThreads,
                          TileLayout::SwizzledRowMajor, TileLayout::RowMajor> {
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

    static const int SmemCacheLineWidth = 128;  // 128B
    static const int crosswise = SmemCacheLineWidth * 8 / kElmentBits / 2;
    using SLayout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<Element>::value, crosswise>;
    using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<kRow, kCol>, Element, SLayout, 1, ThreadMap>;

    using GTileShape = cutlass::MatrixShape<kRow, kCol>;
    using GLayout = cutlass::layout::RowMajor;
    using GmemIterator =
        cutlass::transform::threadblock::PredicatedTileIterator<
            GTileShape, Element, GLayout, 1, ThreadMap>;

    using Fragment = typename GmemIterator::Fragment;

    SmemTileTransmitter(int64_t row_size, int64_t col_size)
        : row_size(row_size), col_size(col_size) {}

    __device__ void transfer(const Element* src, Element* trg, int stride,
                             int tid) const {
        typename SmemIterator::TensorRef ref(
            {const_cast<Element*>(src), SmemIterator::Layout(col_size)});
        SmemIterator smem_iterator(ref, tid);

        typename GmemIterator::Params params({stride});
        GmemIterator gmem_iterator(params, trg, {row_size, col_size}, tid);

        Fragment frag;
        frag.clear();
        smem_iterator.load(frag);
        gmem_iterator.store(frag);
        __syncthreads();
    };

    int64_t row_size;
    int64_t col_size;
};

/// shared memory SwizzledColumnMajor to global memory RowMajor
template <const int kRow, const int kCol, typename Element_, const int kThreads>
class SmemTileTransmitter<kRow, kCol, Element_, kThreads,
                          TileLayout::SwizzledColumnMajor,
                          TileLayout::RowMajor> {
   public:
    using Element = Element_;

    static const int kAccessInBits = 128;
    static const int kElmentBits = cutlass::sizeof_bits<Element>::value;
    static const int kNumPerAccess = kAccessInBits / kElmentBits;

    // warp arrangement: 4 x 8, 8 x 4, 8 x 8
    static const int kWarpContiguous = 4;
    static const int kWarpStride = 32 / kWarpContiguous;

    using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<
        cutlass::layout::PitchLinearShape<kRow, kCol>, kThreads,
        cutlass::layout::PitchLinearShape<kWarpContiguous, kWarpStride>,
        kNumPerAccess>;

    static const int SmemCacheLineWidth = 128;  // 128B
    static const int crosswise = SmemCacheLineWidth * 8 / kElmentBits / 2;
    using SLayout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        cutlass::sizeof_bits<Element>::value, crosswise>;
    using SmemIterator = cutlass::transform::threadblock::RegularTileIterator<
        cutlass::MatrixShape<kRow, kCol>, Element, SLayout, 1, ThreadMap>;

    using GTileShape = cutlass::MatrixShape<kCol, kRow>;
    using GLayout = cutlass::layout::RowMajor;
    using GmemIterator =
        cutlass::transform::threadblock::PredicatedTileIterator<
            GTileShape, Element, GLayout, 1, ThreadMap>;

    using Fragment = typename GmemIterator::Fragment;

    SmemTileTransmitter(int64_t row_size, int64_t col_size)
        : row_size(row_size), col_size(col_size) {}

    __device__ void transfer(const Element* src, Element* trg, int stride,
                             int tid) const {
        typename SmemIterator::TensorRef ref(
            {const_cast<Element*>(src), SmemIterator::Layout(row_size)});
        SmemIterator smem_iterator(ref, tid);

        typename GmemIterator::Params params({stride});
        GmemIterator gmem_iterator(params, trg, {col_size, row_size}, tid);

        Fragment frag;
        frag.clear();
        smem_iterator.load(frag);
        gmem_iterator.store(frag);
        __syncthreads();
    };

    int64_t row_size;
    int64_t col_size;
};

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
