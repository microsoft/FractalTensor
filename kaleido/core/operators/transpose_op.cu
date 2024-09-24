#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/print_op.h"
#include "kaleido/core/operators/transpose_op.h"

#include <glog/logging.h>

namespace kaleido {
namespace core {
namespace ops {

namespace {

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

void CheckInput(const Tensor& input, const Tensor& output,
                const std::vector<size_t> dims) {
    if (input.ndim() != output.ndim() || input.ndim() != dims.size() ||
        output.ndim() != dims.size()) {
        LOG(FATAL)
            << "Expect the dimensions of input tensor, output tensor and "
               "size of dims to be consistent, got "
            << input.ndim() << ", " << output.ndim() << ", " << dims.size()
            << ".";
    }
    std::vector<size_t> temp(dims);
    std::sort(temp.begin(), temp.end());
    for (int i = 1; i < temp.size(); ++i) {
        if (temp[i] == temp[i - 1])
            LOG(FATAL) << "Expect each number in dims to be unique.";
    }
    if (temp[0] != 0 || temp.back() != dims.size() - 1) {
        LOG(FATAL)
            << "Can't align the dimensions of dims with the tensor dimensions.";
    }

    for (int i = 0; i < input.ndim(); ++i) {
        size_t d = output.dim_size(i);
        if (d == 0)
            LOG(FATAL) << "Do not support tensor dimensions with size 0.";
        if (input.dim_size(dims[i]) != d)
            LOG(FATAL)
                << "The output shape should be the same as the tansposed "
                   "input shape.";
    }
}

void GetStride(const Tensor& input, const Tensor& output,
               const std::vector<size_t> dims, std::vector<int64_t>& stride,
               std::vector<double>& stride_inv) {
    std::vector<int64_t> out_stride;
    stride.push_back((int64_t)1);
    out_stride.push_back((int64_t)1);
    for (int i = dims.size() - 1; i > 0; --i) {
        stride.insert(stride.begin(), stride[0] * input.dim_size(i));
        out_stride.insert(out_stride.begin(),
                          out_stride[0] * output.dim_size(i));
    }

    for (auto s : stride) stride_inv.push_back((double)(1.0 / s));

    /* transpose output stride to align with input index. */
    stride.insert(stride.end(), out_stride.begin(), out_stride.end());
    for (int i = 0; i < dims.size(); ++i)
        stride[dims.size() + dims[i]] = out_stride[i];
}

template <typename T>
__device__ void CopyToPosition(const T* in_data, T* out_data,
                               const int64_t* stride, const int64_t idx,
                               const int64_t dim) {
    int64_t out_idx = 0;
    int64_t index = idx;
    for (int i = 0; i < dim; ++i) {
        int64_t id = index / stride[i];
        out_idx += stride[i + dim] * id;
        index = index - id * stride[i];
    }
    out_data[out_idx] = in_data[idx];
}

template <typename T>
__global__ void TransposeKernel(const T* in, T* out, const int64_t* stride,
                                const int64_t numel, const size_t dim,
                                const int round) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx = tid_x;
    for (int i = 0; i < round; ++i) {
        CopyToPosition<T>(in, out, stride, idx, dim);
        idx += gridDim.x * blockDim.x;
    }
    if (idx < numel) { /* tails */
        CopyToPosition<T>(in, out, stride, idx, dim);
    }
}

template <typename T>
__global__ void TransposeKernel2DNoConflict(const T* in, T* out) {
    __shared__ T tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        out[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
}

}  // namespace

template <typename T>
class TransposeOp<GPUContext, CUDAPlace, T> {
   public:
    void operator()(const Tensor& input, Tensor& output,
                    std::vector<size_t> dims) {
        CheckInput(input, output, dims);

        // temp settings
        int block_size = 1024;
        int grid_size = 1024;

        /* compute stride/transposed stride for input/output and (stride_inv).
         */
        int64_t* stride;
        std::vector<int64_t> stride_host;
        std::vector<double> stride_inv_host;
        GetStride(input, output, dims, stride_host, stride_inv_host);

        cudaMalloc(&stride, dims.size() * sizeof(int64_t) * 2);
        cudaMemcpy(stride, stride_host.data(),
                   dims.size() * sizeof(int64_t) * 2, cudaMemcpyHostToDevice);
        const T* in_data = input.data<T>();
        T* out_data = output.mutable_data<T>();

        int round = floor(input.numel() / (grid_size * block_size));
        if (dims.size() == 2) {
            dim3 dimGrid(input.dim_size(0) / TILE_DIM,
                         input.dim_size(1) / TILE_DIM, 1);
            dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
            TransposeKernel2DNoConflict<T>
                <<<dimGrid, dimBlock>>>(in_data, out_data);
        } else {
            TransposeKernel<T><<<grid_size, block_size>>>(
                in_data, out_data, stride, input.numel(), dims.size(), round);
        }
        cudaFree(stride);
    }
};

template class TransposeOp<GPUContext, CUDAPlace, float>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
