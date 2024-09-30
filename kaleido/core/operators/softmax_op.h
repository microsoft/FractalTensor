#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class SoftmaxOp {
   public:
    void operator()(const DeviceContext& context, const Tensor& x, Tensor& y,
                    int dim);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
