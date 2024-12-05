#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class ConcatOp {
   public:
    void operator()(const DeviceContext& context,
                    const std::vector<Tensor>& inputs, Tensor& output,
                    size_t dim);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
