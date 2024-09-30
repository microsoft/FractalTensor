#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/gather_scatter.h"
#include "kaleido/core/operators/gather_nd_op.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename T>
class GatherNdOp<GPUContext, CUDAPlace, T> {
   public:
    void operator()(const GPUContext& context, Tensor& output,
                    const Tensor& input, const Tensor& indices) {
        auto index_dims = indices.dims();
        size_t index_dims_size = indices.ndim();
        auto input_dims = input.dims();
        size_t input_dims_size = input.ndim();

        // indices for the first `end_size` dimensionalities are specified
        int64_t end_size = index_dims[index_dims_size - 1];

        int64_t remain_numel = 1;
        for (int i = 0; i < index_dims_size - 1; ++i)
            remain_numel *= index_dims[i];

        // slice size
        int64_t slice_size = 1;
        for (int64_t i = end_size; i < input_dims_size; ++i) {
            // innermost dimensionalities form contiguous memory to slice.
            slice_size *= input_dims[i];
        }

        int64_t* g_input_dims;
        CudaCheck(cudaMalloc(&g_input_dims, input_dims_size * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(g_input_dims, input_dims.data(),
                             input_dims_size * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

        int64_t block = 512;
        int64_t n = slice_size * remain_numel;
        int64_t grid = (n + block - 1) / block;

        cuda_kernel::GatherNdCUDAKernel<T><<<grid, block, 0>>>(
            input.data<T>(), g_input_dims, indices.data<int64_t>(),
            output.mutable_data<T>(), remain_numel, slice_size, end_size);

        cudaFree(g_input_dims);
    }
};

template class GatherNdOp<GPUContext, CUDAPlace, float>;
template class GatherNdOp<GPUContext, CUDAPlace, int>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
