#pragma once

#include "kaleido/core/tensor.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename DeviceContext, typename Place, typename T>
class FillOp {
   public:
    void operator()(Tensor& input);
    void operator()(Tensor& input, float value);
    void operator()(Tensor& input, float mean, float stddev);
    void operator()(Tensor& input, const std::string& mode, float scale = 1.);
};

}  // namespace ops
}  // namespace core
}  // namespace kaleido
