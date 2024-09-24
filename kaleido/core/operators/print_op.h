#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class PrintOp {
   public:
    std::string operator()(const Tensor& input, int precision = 3,
                           int count = -1, int pos = -1) const;
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
