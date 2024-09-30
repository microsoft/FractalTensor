#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class TransposeOp {
   public:
    void operator()(const Tensor& input, Tensor& output,
                    std::vector<size_t> dims);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
