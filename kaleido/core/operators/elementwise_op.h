#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

enum ElementwiseType {
    kUnary = 1,
    kBinary = 2,
    kTernary = 3,
    kArityFour = 4,
    kAny = -1  //
};

template <typename DeviceContext, typename Place, ElementwiseType ET,
          typename T, typename Functor>
class ElementwiseOp {
   public:
    void operator()(const DeviceContext& context,
                    const std::vector<Tensor>& inputs, Tensor& output,
                    Functor func);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
