#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/elementwise.h"
#include "kaleido/core/device/kernels/math_functor.h"
#include "kaleido/core/operators/elementwise_op.h"
#include "kaleido/core/operators/launch_config.h"

#include <glog/logging.h>

namespace kaleido {
namespace core {
namespace ops {

template <ElementwiseType ET, typename T, typename Functor>
class ElementwiseOp<GPUContext, CUDAPlace, ET, T, Functor> {
   public:
    void operator()(const GPUContext& context,
                    const std::vector<Tensor>& inputs, Tensor& output,
                    Functor func) {
        const int kArity = static_cast<int>(ET);
        if (kArity == -1) LOG(FATAL) << "Not implemented yet.";

        int num_inputs = kArity == -1 ? inputs.size() : kArity;
        int64_t numel = inputs[0].numel();

        int threads;
        int blocks;
        GetGpuLaunchConfig1D(context, numel, &threads, &blocks);

        dim3 block_dims = dim3(threads, 1, 1);
        dim3 grid_dims = dim3(blocks, 1, 1);

        std::vector<const T*> inputs_data_vec(kArity);
        const T** inputs_data = inputs_data_vec.data();
        for (int i = 0; i < kArity; ++i) inputs_data[i] = inputs[i].data<T>();

        const T** dev_inputs_data = nullptr;
        CudaCheck(cudaMalloc(&dev_inputs_data, kArity * sizeof(T*)));
        CudaCheck(cudaMemcpy(dev_inputs_data, inputs_data, kArity * sizeof(T*),
                             cudaMemcpyHostToDevice));

        cuda_kernel::ElementwiseKernel<T, Functor, kArity>
            <<<grid_dims, block_dims, 0>>>(dev_inputs_data,
                                           output.mutable_data<T>(), numel,
                                           func, num_inputs);

        cudaFree(dev_inputs_data);
    }
};

template class ElementwiseOp<GPUContext, CUDAPlace, ElementwiseType::kUnary,
                             float, cuda_kernel::Log<float>>;
template class ElementwiseOp<GPUContext, CUDAPlace, ElementwiseType::kUnary,
                             float, cuda_kernel::Exp<float>>;
template class ElementwiseOp<GPUContext, CUDAPlace, ElementwiseType::kBinary,
                             float, cuda_kernel::Add<float>>;
template class ElementwiseOp<GPUContext, CUDAPlace, ElementwiseType::kBinary,
                             float, cuda_kernel::Sub<float>>;
template class ElementwiseOp<GPUContext, CUDAPlace, ElementwiseType::kBinary,
                             float, cuda_kernel::Multiply<float>>;
template class ElementwiseOp<GPUContext, CUDAPlace, ElementwiseType::kBinary,
                             float, cuda_kernel::Div<float>>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
