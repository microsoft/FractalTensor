#include "kaleido/core/device/cuda_info.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/concat_op.h"

#include <glog/logging.h>

namespace kaleido {
namespace core {
namespace ops {

namespace {
void checkShape(const std::vector<Tensor>& inputs, Tensor& output, size_t dim) {
    /* shape check can be moved into compile-time check. */
    size_t num_input = inputs.size();
    if (!num_input) return;

    auto check_shape = [&, dim](auto s1, auto s2) {
        if (s1.ndim() != s2.ndim()) return false;
        for (size_t i = 0; i < s1.ndim(); ++i) {
            if (i == dim) continue;
            if (s1.dim_size(i) != s2.dim_size(i)) return false;
        }
        return true;
    };

    int total_size = inputs[0].dim_size(dim);
    int ndim = inputs[0].ndim();
    for (int i = 1; i < num_input; ++i) {
        if (!check_shape(inputs[0], inputs[i]))
            LOG(FATAL) << "Except the concatenation dim, "
                       << "all the other dimension should have the same size.";
        total_size += inputs[i].dim_size(dim);
    }

    if (output.ndim() != ndim) LOG(FATAL) << "Error output shape.";
    for (size_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (output.dim_size(i) != total_size)
                LOG(FATAL) << "Error output dimension, expected  " << total_size
                           << ", got " << output.dim_size(i) << ".";
            continue;
        } else {
            if (output.dim_size(i) != inputs[0].dim_size(i))
                LOG(FATAL) << "Error output shape.";
        }
    }
}

static inline void GetBlockDims(const GPUContext& context, int64_t num_rows,
                                int64_t num_cols, dim3* block_dims,
                                dim3* grid_dims) {
    const int kThreadsPerBlock = context.GetMaxThreadsPerBlock();
    int block_cols = kThreadsPerBlock;
    if (num_cols < kThreadsPerBlock)
        // integer division to align with 32.
        block_cols = ((num_cols + 31) >> 5) << 5;
    int block_rows = kThreadsPerBlock / block_cols;
    *block_dims = dim3(block_cols, block_rows, 1);

    int max_threads = context.GetMaxPhysicalThreadCount();
    int64_t max_blocks = std::max(max_threads / kThreadsPerBlock, 1);

    int grid_cols =
        std::min((num_cols + block_cols - 1) / block_cols, max_blocks);
    int grid_rows = std::min(max_blocks / grid_cols,
                             std::max(num_rows / block_rows, (int64_t)1));
    *grid_dims = dim3(grid_cols, grid_rows, 1);
}

template <typename T>
__global__ void ConcatKernel(const T** inputs, const int64_t* input_cols,
                             int col_size, const int64_t output_rows,
                             const int64_t output_cols, T* output) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_segment = 0;
    int curr_offset = input_cols[0];
    for (; tid_x < output_cols; tid_x += blockDim.x * gridDim.x) {
        int curr_col_offset = input_cols[curr_segment + 1];
        while (curr_col_offset <= tid_x) {
            curr_offset = curr_col_offset;
            ++curr_segment;
            curr_col_offset = input_cols[curr_segment + 1];
        }

        int local_col = tid_x - curr_offset;
        int segment_width = curr_col_offset - curr_offset;

        const T* input_ptr = inputs[curr_segment];
        int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
        for (; tid_y < output_rows; tid_y += blockDim.y * gridDim.y)
            output[tid_y * output_cols + tid_x] =
                input_ptr[tid_y * segment_width + local_col];
    }
}

template <typename T>
__device__ void ConcatKernelDetail(const T** inputs_data,
                                   const int fixed_in_col, const int out_rows,
                                   const int out_cols, T* output_data) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid_x < out_cols; tid_x += blockDim.x * gridDim.x) {
        int split = tid_x * 1.0 / fixed_in_col;
        int in_offset = tid_x - split * fixed_in_col;
        const T* input_ptr = inputs_data[split];
        int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
        for (; tid_y < out_rows; tid_y += blockDim.y * gridDim.y) {
            output_data[tid_y * out_cols + tid_x] =
                input_ptr[tid_y * fixed_in_col + in_offset];
        }
    }
}

template <typename T>
__global__ void ConcatKernel(const T* input_addr0, const T* input_addr1,
                             const int64_t fixed_in_col, const int64_t out_rows,
                             const int64_t out_cols, T* output_data) {
    const T* inputs_data[2];
    inputs_data[0] = input_addr0;
    inputs_data[1] = input_addr1;
    ConcatKernelDetail<T>(inputs_data, fixed_in_col, out_rows, out_cols,
                          output_data);
}

template <typename T>
__global__ void ConcatKernel(const T* input_addr0, const T* input_addr1,
                             const T* input_addr2, const int64_t fixed_in_col,
                             const int64_t out_rows, const int64_t out_cols,
                             T* output_data) {
    const T* inputs_data[3];
    inputs_data[0] = input_addr0;
    inputs_data[1] = input_addr1;
    inputs_data[2] = input_addr2;
    ConcatKernelDetail<T>(inputs_data, fixed_in_col, out_rows, out_cols,
                          output_data);
}

template <typename T>
__global__ void ConcatKernel(const T* input_addr0, const T* input_addr1,
                             const T* input_addr2, const T* input_addr3,
                             const int64_t fixed_in_col, const int64_t out_rows,
                             const int64_t out_cols, T* output_data) {
    const T* inputs_data[4];
    inputs_data[0] = input_addr0;
    inputs_data[1] = input_addr1;
    inputs_data[2] = input_addr2;
    inputs_data[3] = input_addr3;
    ConcatKernelDetail<T>(inputs_data, fixed_in_col, out_rows, out_cols,
                          output_data);
}

template <typename T>
__global__ void ConcatKernel(const T** inputs_data, const int in_num,
                             const int64_t fixed_in_col, const int64_t out_rows,
                             const int64_t out_cols, T* output_data) {
    ConcatKernelDetail<T>(inputs_data, fixed_in_col, out_rows, out_cols,
                          output_data);
}

}  // namespace

template <typename T>
class ConcatOp<GPUContext, CUDAPlace, T> {
   public:
    void operator()(const GPUContext& context,
                    const std::vector<Tensor>& inputs, Tensor& output,
                    size_t dim) {
        checkShape(inputs, output, dim);

        // flatten the `0`-th to the `dim`-th dimensionality as row.
        int64_t in_row = 1;
        for (size_t i = 0; i < dim; ++i) in_row *= inputs[0].dim_size(i);
        int64_t in_col = inputs[0].numel() / in_row;
        int64_t out_row = in_row, out_col = 0;

        size_t num_input = inputs.size();
        int64_t inputs_col_num = num_input + 1;
        std::vector<const T*> inputs_data_vec(num_input);
        std::vector<int64_t> inputs_col_vec(inputs_col_num);
        const T** inputs_data = inputs_data_vec.data();
        int64_t* inputs_col = inputs_col_vec.data();

        inputs_col[0] = 0;
        // indicate whether inputs to concatenate has the same shape
        bool has_same_shape = true;
        for (int i = 0; i < num_input; ++i) {
            int64_t t_cols = inputs[i].numel() / in_row;
            if (has_same_shape) {
                if (t_cols != in_col) has_same_shape = false;
            }
            out_col += t_cols;
            inputs_col[i + 1] = out_col;
            inputs_data[i] = inputs[i].data<T>();
        }

        dim3 block_dims;
        dim3 grid_dims;

        GetBlockDims(context, out_row, out_col, &block_dims, &grid_dims);

        const T** dev_inputs_data = nullptr;
        if (!has_same_shape || num_input < 2 || num_input > 4) {
            CudaCheck(cudaMalloc(&dev_inputs_data, num_input * sizeof(T*)));
            CudaCheck(cudaMemcpy(dev_inputs_data, inputs_data,
                                 num_input * sizeof(T*),
                                 cudaMemcpyHostToDevice));
        }

        if (has_same_shape) {
            if (num_input == 2) {
                ConcatKernel<<<grid_dims, block_dims, 0>>>(
                    inputs_data[0], inputs_data[1], in_col, out_row, out_col,
                    output.mutable_data<T>());
            } else if (num_input == 3) {
                ConcatKernel<<<grid_dims, block_dims, 0>>>(
                    inputs_data[0], inputs_data[1], inputs_data[2], in_col,
                    out_row, out_col, output.mutable_data<T>());
            } else if (num_input == 4) {
                ConcatKernel<<<grid_dims, block_dims, 0>>>(
                    inputs_data[0], inputs_data[1], inputs_data[2],
                    inputs_data[3], in_col, out_row, out_col,
                    output.mutable_data<T>());
            } else {
                ConcatKernel<<<grid_dims, block_dims, 0>>>(
                    dev_inputs_data, num_input, in_col, out_row, out_col,
                    output.mutable_data<T>());
                cudaFree(dev_inputs_data);
            }
        } else {
            int64_t* dev_inputs_col_data = nullptr;
            CudaCheck(cudaMalloc(&dev_inputs_col_data,
                                 inputs_col_num * sizeof(int64_t)));
            CudaCheck(cudaMemcpy(dev_inputs_col_data, inputs_col,
                                 inputs_col_num * sizeof(int64_t),
                                 cudaMemcpyHostToDevice));

            ConcatKernel<<<grid_dims, block_dims, 0>>>(
                dev_inputs_data, dev_inputs_col_data,
                static_cast<int>(inputs_col_num), out_row, out_col,
                output.mutable_data<T>());

            cudaFree(dev_inputs_col_data);
            cudaFree(dev_inputs_data);
        }
    }
};

template class ConcatOp<GPUContext, CUDAPlace, float>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
