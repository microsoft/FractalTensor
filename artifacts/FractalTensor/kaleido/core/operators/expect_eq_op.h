#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class ExpectEqOp {
   public:
    void operator()(const Tensor& x, const Tensor& y, float epsilon = 1e-5);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
