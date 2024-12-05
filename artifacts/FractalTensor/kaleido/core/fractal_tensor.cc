#include "kaleido/core/fractal_tensor.h"

#include <iostream>

namespace kaleido {
namespace core {

std::string FractalTensor::DebugString() const {
    return type_desc_.DebugString();
}

}  // namespace core
}  // namespace kaleido
