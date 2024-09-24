#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/softmax.h"
#include "kaleido/core/operators/softmax_op.h"
#include "kaleido/core/types.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename T>
class SoftmaxOp<GPUContext, CUDAPlace, T> {
   public:
    void operator()(const GPUContext& context, const Tensor& x, Tensor& y,
                    int dim = 0) {
        assert(x.ndim() == 2 && x.ndim() == 2 && x.shape() == y.shape());
        if (dim == 1) LOG(FATAL) << "Not implmented yet.";

        const int kThreadsPerBlock = context.GetMaxThreadsPerBlock();
        int width = x.dim_size(1);
        int height = x.dim_size(0);

        int block_num =
            width > kThreadsPerBlock
                ? kThreadsPerBlock
                : pow(2, static_cast<int>(log2(static_cast<float>(width))));

        dim3 block(block_num, 1);
        dim3 grid(height, 1);

        cuda_kernel::KeMatrixSoftMax<<<grid, block, 0>>>(
            x.data<T>(), y.mutable_data<T>(), width);
    }
};

template class SoftmaxOp<GPUContext, CUDAPlace, float>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
