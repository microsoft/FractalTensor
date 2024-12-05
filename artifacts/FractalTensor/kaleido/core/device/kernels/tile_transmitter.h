#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>
#include <cutlass/aligned_buffer.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/pitch_linear.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/pitch_linear_thread_map.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h>

namespace kaleido {
namespace core {
namespace cuda_kernel {

enum class TileLayout {
    RowMajor = 0,
    ColumnMajor = 1,
    SwizzledRowMajor = 2,  // shared memory layout
    SwizzledColumnMajor = 3
};

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
