#pragma once

namespace access_kernel {

__global__ void KeNaiveSelectRowsKernel(float* O, const float* I,
                                        const int64_t* rows, int height,
                                        int width) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (threadIdx.x < width && row_idx < height) {
        int64_t to_pos = row_idx * width + threadIdx.x;
        int64_t from_pos = rows[row_idx] * width + threadIdx.x;
        O[to_pos] = I[from_pos];
    }
}

__global__ void KeMiddleIds(int64_t* ids, int bs, int len, int blk_num,
                            int blksz, int gs, int middle_blk_num) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_x < bs * middle_blk_num * blksz) {
        int B = int(tid_x / (middle_blk_num * blksz));
        int R = int((tid_x - B * middle_blk_num * blksz) / blksz);
        int S = tid_x - B * middle_blk_num * blksz - R * blksz;
        int64_t write_val = B * len + (R + gs) * blksz + S;
        ids[tid_x] = write_val;
    }
}

__global__ void KeWriteRandomIds(int64_t* ids, int64_t* rand_attn, int ws,
                                 int blksz, int rs, int len,
                                 int middle_block_num, int bs) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_x < bs * middle_block_num * rs * blksz) {
        int row_size = rs * blksz;
        int batch_size = middle_block_num * row_size;
        int B = tid_x / batch_size;
        int R = (tid_x - B * batch_size) / row_size;
        int S = tid_x - (B * middle_block_num + R) * row_size;

        int64_t write_value = B * len + rand_attn[R * rs + S / blksz];
        ids[tid_x] = write_value + S - S / blksz;
    }
}

__global__ void KeSetStridedContinuous(int64_t* ids, int start, int batch_width,
                                       int input_stride, int row_size) {
    int64_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_x < row_size) {
        int batch_idx = tid_x / batch_width;
        int x = tid_x - batch_idx * batch_width;
        ids[tid_x] = start + batch_idx * input_stride + x;
    }
}

__global__ void KeNaiveScatterRowsKernelStrided(float* O, const float* I,
                                                const int64_t* rows, int height,
                                                int width, int output_width) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (threadIdx.x < width && row_idx < height) {
        int64_t to_pos = row_idx * width + threadIdx.x;
        int64_t from_pos = rows[row_idx] * output_width + threadIdx.x;
        O[from_pos] = I[to_pos];
    }
}

__global__ void KeNaiveGatherRowsKernelStrided(float* O, const float* I,
                                               const int64_t* rows, int height,
                                               int width, int input_width) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (threadIdx.x < width && row_idx < height) {
        int64_t to_pos = row_idx * width + threadIdx.x;
        int64_t from_pos = rows[row_idx] * input_width + threadIdx.x;
        O[to_pos] = I[from_pos];
    }
}

__global__ void KeNaiveScatterRowsKernel(float* O, const float* I,
                                         const int64_t* rows, int height,
                                         int width) {
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (threadIdx.x < width && row_idx < height) {
        int64_t to_pos = row_idx * width + threadIdx.x;
        int64_t from_pos = rows[row_idx] * width + threadIdx.x;
        O[from_pos] = I[to_pos];
    }
}

__global__ void KeSetContinuous(int64_t* ids, int64_t start,
                                int64_t middle_row_size) {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_x < middle_row_size) ids[tid_x] = start + tid_x;
}

}  // namespace access_kernel
