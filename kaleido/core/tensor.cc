// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/tensor.h"

#include <iostream>

namespace kaleido {
namespace core {

std::string Tensor::DebugString() const {
  std::stringstream ss;
  ss << "Tensor {" << std::endl << type_desc_.DebugString() << std::endl << "}";
  return ss.str();
}

}  // namespace core
}  // namespace kaleido
