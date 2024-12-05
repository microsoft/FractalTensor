#pragma once

#include "kaleido/core/tensor.h"

#include <vector>

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class GemmBatchedOp {
   public:
    void operator()(const DeviceContext& context, const std::vector<Tensor>& A,
                    bool trans_a, const std::vector<Tensor>& B, bool trans_b,
                    std::vector<Tensor>& C, T alf = 1., T bet = 0.);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
