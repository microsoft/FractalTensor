#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class MatMulOp {
   public:
    void operator()(const DeviceContext& context, const Tensor& A, bool trans_a,
                    const Tensor& B, bool trans_b, Tensor& C, T alf = 1.,
                    T bet = 0.);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
