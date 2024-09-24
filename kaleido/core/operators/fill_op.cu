#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/fill.h"
#include "kaleido/core/operators/fill_op.h"

#include <cutlass/numeric_types.h>

namespace kaleido {
namespace core {
namespace ops {

template <typename T>
class FillOp<GPUContext, CUDAPlace, T> {
   public:
    void operator()(Tensor& input, float value) {
        int numel = static_cast<int>(input.numel());
        T* data = input.mutable_data<T>();

        int threads = 128;
        int blocks = DIVUP(numel, threads);
        cuda_kernel::KeFillValue<T><<<blocks, threads>>>(data, numel, value);
    }

    void operator()(Tensor& input) {
        T* data = input.mutable_data<T>();
        int num = static_cast<int>(input.numel());
        cuda_kernel::FillRandomValue<T>(data, num);
    }

    void operator()(Tensor& input, float mean = 0, float stddev = 0.1) {
        T* data = input.mutable_data<T>();
        int num = static_cast<int>(input.numel());
        cuda_kernel::FillRandomValue<T>(data, num, mean, stddev);
    }

    void operator()(Tensor& input, const std::string& mode, float scale = 1.) {
        if (mode == "seq") {
            T* data = input.mutable_data<T>();
            int64_t numel = input.numel();

            int threads = 128;
            int blocks = DIVUP(numel, threads);
            cuda_kernel::KeFillSequential<T>
                <<<blocks, threads>>>(data, numel, scale);
        } else {
            LOG(FATAL) << "Unknown mode: " << mode << std::endl;
        }
    }
};

template class FillOp<GPUContext, CUDAPlace, float>;
template class FillOp<GPUContext, CUDAPlace, __half>;
template class FillOp<GPUContext, CUDAPlace, cutlass::half_t>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
