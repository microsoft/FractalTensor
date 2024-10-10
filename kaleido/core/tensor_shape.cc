// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/tensor_shape.h"

namespace kaleido {
namespace core {

bool TensorShape::IsEuqalShape(const TensorShape& b) const {
    if (b.ndim() != ndim()) return false;
    for (size_t i = 0; i < ndim(); ++i) {
        if (b.dim_size(i) != dim_size(i)) return false;
    }
    return true;
}

std::string TensorShape::DebugString() const {
    std::stringstream ss;
    ss << "shape : [";
    for (size_t i = 0; i < dim_ - 1; ++i) ss << dim_sizes_[i] << ", ";
    ss << dim_sizes_[dim_ - 1] << "]";
    return ss.str();
}

}  // namespace core
}  // namespace kaleido
