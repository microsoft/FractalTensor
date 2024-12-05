#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class GatherNdOp {
   public:
    void operator()(const DeviceContext& context, Tensor& output,
                    const Tensor& input, const Tensor& indices);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
