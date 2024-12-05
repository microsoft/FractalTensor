#pragma once

#include <cute/layout.hpp>

namespace kaleido {
namespace core {

using namespace cute;

// In the row major layout, the contiguous dimension in memory is the
// last dimension.
template <const int row, const int col, const int stride = col>
using RowMajor =
    cute::Layout<Shape<Int<row>, Int<col>>, Stride<Int<stride>, _1>>;

__device__ auto make_row_major_layout(const int row, const int col,
                                      const int stride) {
    return cute::make_layout(make_shape(row, col), make_stride(stride, 1));
}

// In the column major layout, the contiguous dimension in memory is the
// first dimension.
template <const int row, const int col, const int stride = row>
using ColMajor =
    cute::Layout<Shape<Int<row>, Int<col>>, Stride<_1, Int<stride>>>;

__device__ auto make_col_major_layout(const int row, const int col,
                                      const int stride) {
    return cute::make_layout(make_shape(row, col), make_stride(1, stride));
}

template <typename Layout>
static constexpr size_t num_rows = cute::size<0>(Layout{});

template <typename Layout> /*  */
static constexpr size_t num_cols = cute::size<1>(Layout{});

}  // namespace core
}  // namespace kaleido
