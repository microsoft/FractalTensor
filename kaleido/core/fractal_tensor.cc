// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/fractal_tensor.h"

#include <iostream>

namespace kaleido {
namespace core {

std::string FractalTensor::DebugString() const {
  return type_desc_.DebugString();
}

}  // namespace core
}  // namespace kaleido
