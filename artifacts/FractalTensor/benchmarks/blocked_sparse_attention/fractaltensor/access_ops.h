#pragma once

#include "access_ops_kernels.cuh"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/kernels/softmax.h"

#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

using namespace kaleido::core;

void init_data(float* Q_d, float* K_d, float* V_d, float* Res_d, int bs,
               int len, int h) {
    int64_t numel = bs * len * h;
    float* Q_h = (float*)malloc(numel * sizeof(float));
    float* K_h = (float*)malloc(numel * sizeof(float));
    float* V_h = (float*)malloc(numel * sizeof(float));
    float* Res_h = (float*)malloc(numel * sizeof(float));
    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < len * h; ++j) {
            Q_h[i * len * h + j] = float(j);
            K_h[i * len * h + j] = j < 2 * h ? float(1) : float(0);
            V_h[i * len * h + j] = float(1);
            Res_h[i * len * h + j] = float(0);
        }
    }
    CudaCheck(
        cudaMemcpy(Q_d, Q_h, numel * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(
        cudaMemcpy(K_d, K_h, numel * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(
        cudaMemcpy(V_d, V_h, numel * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(Res_d, Res_h, numel * sizeof(float),
                         cudaMemcpyHostToDevice));
}

/* generate r random values in [global_size, window_start) U (window_end,
 * len-global_size]. */
void GetRowRandPosition(const int64_t window_start, const int64_t window_end,
                        const int64_t len, const int64_t r,
                        std::vector<int64_t>& row) {
    std::unordered_set<int64_t> rand_values;
    std::random_device rd;
    std::mt19937 s(rd());
    // window length may be different at different row.
    std::uniform_int_distribution<> random_gen(
        1, len - (window_end - window_start) - 1);

    while (rand_values.size() < r) {
        rand_values.insert(random_gen(s));
    }
    for (auto v : rand_values) {
        row.emplace_back((v + window_end) % len);
    }
}

void GenerateRandAttn(const int64_t block_num, const int64_t num_rand_blocks,
                      const int64_t window_size, const int64_t global_size,
                      std::vector<int64_t>& rand) {
    for (int64_t i = 0; i < block_num - 2; ++i) {
        int64_t w = (window_size - 1) / 2;
        int64_t window_start = std::max(int64_t(0), i - w);
        int64_t window_end = std::min(block_num - 2, i + w);
        std::vector<int64_t> row;
        GetRowRandPosition(window_start, window_end, block_num - 2,
                           num_rand_blocks, row);
        for (auto s : row) rand.emplace_back(s /* + global_size*/);
    }
}

void init_rand_attn(int64_t len, int64_t blksz, int64_t rs, int64_t ws,
                    int64_t gs, std::vector<int64_t>& rand_attn_pos) {
    GenerateRandAttn(len / blksz, rs, 2 * ws + 1, gs, rand_attn_pos);
}

namespace access_ops {

inline void push_block_rows(std::vector<int64_t>& ids, int64_t start,
                            int64_t len) {
    for (int64_t i = 0; i < len; ++i) ids.emplace_back(start + i);
}

void build_dense_row_ids(int64_t* ids, int bs, int len, int gs, int blksz) {
    std::vector<int64_t> dense_row_ids_h;
    for (int b = 0; b < bs; ++b) {
        push_block_rows(dense_row_ids_h, b * len, blksz * gs);
        push_block_rows(dense_row_ids_h, (b + 1) * len - blksz * gs,
                        blksz * gs);
    }
    CudaCheck(cudaMemcpy(ids, dense_row_ids_h.data(),
                         bs * (2 * gs * blksz) * sizeof(int64_t),
                         cudaMemcpyHostToDevice));
}

void gather_attention_rows(const float* input, float* output,
                           const int64_t* rows, int width, int row_num) {
    const int kThreadsPerBlock = 512;

    int block_x = kThreadsPerBlock;
    if (width < kThreadsPerBlock)
        // integer division to align with 32.
        block_x = ((width + 31) >> 5) << 5;
    int block_y = kThreadsPerBlock / block_x;
    dim3 block = dim3(block_x, block_y, 1);

    int grid_x = std::max(row_num / block_y, 1);
    dim3 grid(grid_x, 1);

    access_kernel::KeNaiveSelectRowsKernel<<<grid, block, 0>>>(
        output, input, rows, row_num, width);
}

template <typename T>
void attention_score_softmax_op(const T* in, T* out, size_t width,
                                int64_t height) {
    const int kThreadsPerBlock = 512;
    int block_num =
        width > kThreadsPerBlock
            ? kThreadsPerBlock
            : pow(2, static_cast<int>(log2(static_cast<float>(width))));
    dim3 block(block_num, 1);
    dim3 grid(height, 1);

    cuda_kernel::KeMatrixSoftMax<<<grid, block, 0>>>(in, out, width);
}
template void attention_score_softmax_op(const float* in, float* out,
                                         size_t width, int64_t height);

// return C(row-major) = A * BT
void cublasSgemmStridedBatchedQK(cublasHandle_t handle, const float* A,
                                 const float* B, float* C, int A_row, int B_row,
                                 int A_col, int bs) {
    float alpha = 1.0f;
    float beta = 0.0f;

    // CT = B * AT
    CublasCheck(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, B_row, A_row, A_col, &alpha, B, A_col,
        B_row * A_col, A, A_col, A_row * A_col, &beta, C, B_row, B_row * A_row,
        bs));
}

void cublasSgemmStridedBatchedSV(cublasHandle_t handle, const float* A,
                                 const float* B, float* C, int A_row, int B_col,
                                 int A_col, int bs) {
    float alpha = 1.0f;
    float beta = 0.0f;

    // CT = BT * AT
    // [b, dense_row_size, len] * [b, len, h] -> [b, dense_row_size, h]
    CublasCheck(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, B_col, A_row, A_col, &alpha, B, B_col,
        B_col * A_col, A, A_col, A_col * A_row, &beta, C, B_col, A_row * B_col,
        bs));
}

// return C(row-major) = A * BT (matrix B are overlapped)
void cublasSgemmStridedBatchedQWindowK(cublasHandle_t handle, const float* A,
                                       const float* B, float* C, int A_row,
                                       int B_row, int A_col, int window_stride,
                                       int bs) {
    float alpha = 1.0f;
    float beta = 0.0f;

    // CT = B * AT  blksz, 3*blksz, h,
    CublasCheck(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, B_row, A_row, A_col, &alpha, B, A_col,
        window_stride * A_col /*<--※※B stride*/, A, A_col, A_row * A_col, &beta,
        C, B_row, B_row * A_row, bs));
}

void build_middle_row_ids_gpu(int64_t* ids, int bs, int len, int blk_num,
                              int blksz, int gs, int ws) {
    int middle_blk_num = blk_num - 2 * (gs + ws);
    dim3 block = dim3(1024);
    dim3 grid = dim3(std::max(bs * middle_blk_num * blksz / 1024 + 1, 1));
    access_kernel::KeMiddleIds<<<grid, block>>>(ids, bs, len, blk_num, blksz,
                                                gs, middle_blk_num);
}

void build_middle_row_rand_col_ids_gpu(int64_t* ids, int64_t* rand_attn, int bs,
                                       int len, int blk_num, int blksz, int gs,
                                       int ws, int rs) {
    int middle_block_num = blk_num - 2 * (ws + gs);

    dim3 block_r = dim3(1024);
    dim3 grid_r =
        dim3(std::max(bs * middle_block_num * blksz * rs / 1024 + 1, 1));
    access_kernel::KeWriteRandomIds<<<grid_r, block_r>>>(
        ids, rand_attn, ws, blksz, rs, len, middle_block_num, bs);
}

void build_score_ids(int64_t* ids, int64_t row_size, int in_size, int out_size,
                     int start_offset) {
    std::vector<int64_t> ids_h;
    dim3 block = dim3(1024);
    dim3 grid = dim3(std::max(row_size / 1024 + 1, int64_t(1)));

    access_kernel::KeSetStridedContinuous<<<grid, block>>>(
        ids, start_offset, in_size, out_size, row_size);
}

void scatter_attention_rows_with_stride(const float* input, float* output,
                                        const int64_t* rows, int width,
                                        int output_width, int row_num) {
    const int kThreadsPerBlock = 1024;

    int block_x = kThreadsPerBlock;
    if (width < kThreadsPerBlock)
        // integer division to align with 32.
        block_x = ((width + 31) >> 5) << 5;
    int block_y = kThreadsPerBlock / block_x;
    dim3 block = dim3(block_x, block_y, 1);

    int grid_x = std::max(row_num / block_y, 1);
    dim3 grid(grid_x, 1);

    access_kernel::KeNaiveScatterRowsKernelStrided<<<grid, block, 0>>>(
        output, input, rows, row_num, width, output_width);
}

void gather_attention_rows_with_stride(const float* input, float* output,
                                       const int64_t* rows, int width,
                                       int input_width, int row_num) {
    const int kThreadsPerBlock = 1024;

    int block_x = kThreadsPerBlock;
    if (width < kThreadsPerBlock)
        // integer division to align with 32.
        block_x = ((width + 31) >> 5) << 5;
    int block_y = kThreadsPerBlock / block_x;
    dim3 block = dim3(block_x, block_y, 1);

    int grid_x = std::max(row_num / block_y, 1);
    dim3 grid(grid_x, 1);

    access_kernel::KeNaiveGatherRowsKernelStrided<<<grid, block, 0>>>(
        output, input, rows, row_num, width, input_width);
}

void cublasSgemmStridedBatchedSWindowV(cublasHandle_t handle, const float* A,
                                       const float* B, float* C, int A_row,
                                       int B_col, int A_col, int window_stride,
                                       int bs) {
    float alpha = 1.0f;
    float beta = 0.0f;

    // CT = BT * AT
    // [b, dense_row_size, len] * [b, len, h] -> [b, dense_row_size, h]
    CublasCheck(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, B_col, A_row, A_col, &alpha, B, B_col,
        B_col * window_stride, A, A_col, A_col * A_row, &beta, C, B_col,
        A_row * B_col, bs));
}

void scatter_attention_rows(const float* input, float* output,
                            const int64_t* rows, int width, int row_num) {
    const int kThreadsPerBlock = 1024;

    int block_x = kThreadsPerBlock;
    if (width < kThreadsPerBlock)
        // integer division to align with 32.
        block_x = ((width + 31) >> 5) << 5;
    int block_y = kThreadsPerBlock / block_x;
    dim3 block = dim3(block_x, block_y, 1);

    int grid_x = std::max(row_num / block_y, 1);
    dim3 grid(grid_x, 1);

    access_kernel::KeNaiveScatterRowsKernel<<<grid, block, 0>>>(
        output, input, rows, row_num, width);
}

void build_special_row_ids(int64_t* ids, int idx, int bs, int len, int blk_num,
                           int blksz, int gs, int ws) {
    std::vector<int64_t> ids_h;
    for (int b = 0; b < bs; ++b) {
        push_block_rows(ids_h, b * len + idx * blksz, blksz);
        push_block_rows(ids_h, b * len + len - ((idx + 1) * blksz), blksz);
    }
    cudaMemcpy(ids, ids_h.data(), bs * 2 * blksz * sizeof(int64_t),
               cudaMemcpyHostToDevice);
}

void build_special_cols_ids(int64_t* ids, int idx, int window_size, int bs,
                            int len, int blk_num, int blksz, int gs, int ws,
                            int rs) {
    int64_t row_size = bs * 2 * (gs + rs + window_size) * blksz;
    dim3 block = dim3(1024);
    dim3 grid = dim3(std::max(row_size / 1024 + 1, int64_t(1)));
    access_kernel::KeSetContinuous<<<grid, block>>>(ids, 0, row_size);
}

}  // namespace access_ops
