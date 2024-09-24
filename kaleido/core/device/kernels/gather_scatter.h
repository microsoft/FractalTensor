#pragma once

namespace kaleido {
namespace core {
namespace cuda_kernel {

#define CUDA_KERNEL_LOOP_TYPE(i, num, index_type)              \
    int64_t __index__ = blockIdx.x * blockDim.x + threadIdx.x; \
    for (index_type i = __index__; __index__ < (num);          \
         __index__ += blockDim.x * gridDim.x, i = __index__)

template <typename T>
__global__ void GatherNdCUDAKernel(const T* input, const int64_t* input_dims,
                                   const int64_t* indices, T* output,
                                   size_t remain_size, size_t slice_size,
                                   size_t end_size) {
    CUDA_KERNEL_LOOP_TYPE(i, remain_size * slice_size, int64_t) {
        int64_t indices_i = i / slice_size;
        int64_t slice_i =
            i - indices_i * slice_size;  // offset inside the slice
        int64_t gather_i = 0;
        int64_t temp = slice_size;
        for (int64_t j = end_size - 1; j >= 0; --j) {
            auto index_value = indices[indices_i * end_size + j];
            gather_i += (index_value * temp);
            temp *= input_dims[j];
        }
        int64_t input_i = gather_i + slice_i;
        *(output + i) = *(input + input_i);
    }
}

template <typename T>
__global__ void ScatterNdCUDAKernel(const T* update, const int64_t* indices,
                                    T* output, const int64_t* output_dims,
                                    size_t remain_size, size_t slice_size,
                                    size_t end_size) {
    CUDA_KERNEL_LOOP_TYPE(i, remain_size * slice_size, int64_t) {
        int64_t indices_i = i / slice_size;
        int64_t slice_i =
            i - indices_i * slice_size;  // offset inside the slice
        int64_t gather_i = 0;
        int64_t temp = slice_size;
        for (int64_t j = end_size - 1; j >= 0; --j) {
            int64_t index_value = indices[indices_i * end_size + j];

            gather_i += (index_value * temp);
            temp *= output_dims[j];
        }
        int64_t output_i = gather_i + slice_i;
        atomicAdd(output + output_i, *(update + i));
    }
}

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
