#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class ScatterNdAddOp {
   public:
    void operator()(const DeviceContext& context, Tensor& data,
                    const Tensor& updates, const Tensor& indices);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
