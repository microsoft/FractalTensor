#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/gather_scatter.h"
#include "kaleido/core/operators/scatter_nd_op.h"

namespace kaleido {
namespace core {
namespace ops {

template <typename T>
class ScatterNdAddOp<GPUContext, CUDAPlace, T> {
   public:
    void operator()(const GPUContext& context, Tensor& data,
                    const Tensor& updates, const Tensor& indices) {
        auto index_dims = indices.dims();
        auto index_dims_size = index_dims.size();

        // output_dims = data.dims()
        auto output_dims = data.dims();
        auto output_dims_size = output_dims.size();

        // final dim
        int64_t end_size = index_dims[index_dims_size - 1];

        // remain dim
        auto remain_dims = std::vector<int64_t>(
            index_dims.begin(), index_dims.end() - index_dims_size);

        // Compute the product of the indices dimensions.
        int64_t remain_numel = 1;
        for (int i = 0; i < index_dims_size - 1; ++i)
            remain_numel *= index_dims[i];

        // slice size
        int64_t slice_size = 1;

        // Calculate the product of output dimensions.
        for (int64_t i = end_size; i < output_dims_size; ++i)
            slice_size *= output_dims[i];

        // Calculate bytes of each slice.
        const size_t slice_bytes = slice_size * sizeof(T);

        int64_t* g_output_dims;
        CudaCheck(
            cudaMalloc(&g_output_dims, output_dims_size * sizeof(int64_t)));
        CudaCheck(cudaMemcpy(g_output_dims, output_dims.data(),
                             output_dims_size * sizeof(int64_t),
                             cudaMemcpyHostToDevice));

        int64_t block = 512;
        int64_t n = slice_size * remain_numel;
        int64_t grid = (n + block - 1) / block;

        cuda_kernel::ScatterNdCUDAKernel<T><<<grid, block, 0>>>(
            updates.data<T>(), indices.data<int64_t>(), data.mutable_data<T>(),
            g_output_dims, remain_numel, slice_size, end_size);
    }
};

template class ScatterNdAddOp<GPUContext, CUDAPlace, float>;
template class ScatterNdAddOp<GPUContext, CUDAPlace, int>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
