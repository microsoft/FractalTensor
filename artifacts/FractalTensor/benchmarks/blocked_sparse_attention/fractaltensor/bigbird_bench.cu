#include "access_ops.h"
#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/cuda_utils.h"

using namespace kaleido::core;
using namespace access_ops;

namespace {

void DenseRows(cublasHandle_t& handle, const float* Q, const float* K,
               const float* V, int64_t* dense_row_ids, float* dense_row,
               float* dense_score, float* dense_score_tmp, float* dense_result,
               int len, int h, int bs, int gs, int blksz, int dense_row_size,
               float scal) {
    build_dense_row_ids(dense_row_ids /*on device memory*/, bs, len, gs, blksz);
    // gather first&last block row in one kernel.
    gather_attention_rows(Q, dense_row, dense_row_ids, h, bs * dense_row_size);

    cublasSgemmStridedBatchedQK(handle, dense_row, K, dense_score_tmp,
                                dense_row_size, len, h, bs);
    CublasCheck(cublasSscal_v2(handle, bs * len * dense_row_size, &scal,
                               dense_score_tmp, 1));
    attention_score_softmax_op(dense_score_tmp, dense_score, len,
                               bs * dense_row_size);
    cublasSgemmStridedBatchedSV(handle, dense_score, V, dense_result,
                                dense_row_size, h, len, bs);
}

void MiddleRows(cublasHandle_t& handle, const float* Q, const float* K,
                const float* V, float* Res, float* middle_rows,
                float* global_cols, float* global_cols_v, float* rand_cols,
                float* rand_cols_v, float* global_score, float* rand_score,
                float* middle_score, float* window_score,
                float* middle_score_tmp, float* global_res, float* window_res,
                float* rand_res, float alpha, float scal,
                int64_t* dense_row_ids, int64_t* middle_rows_ids,
                int64_t* rand_cols_ids, int64_t* rand_attn,
                int64_t* scatter_score_ids, int64_t* window_scatter_score_ids,
                int64_t middle_size, int64_t global_col_size,
                int64_t rand_col_size, int64_t window_col_size,
                int64_t middle_block_num, int64_t h, int64_t bs, int64_t len,
                int64_t blk_num, int64_t blksz, int64_t gs, int64_t ws,
                int64_t rs) {
    build_middle_row_ids_gpu(middle_rows_ids, bs, len, blk_num, blksz, gs, ws);
    build_middle_row_rand_col_ids_gpu(rand_cols_ids, rand_attn, bs, len,
                                      blk_num, blksz, gs, ws, rs);
    build_score_ids(scatter_score_ids, bs * middle_size, middle_size, len,
                    blksz);
    build_score_ids(window_scatter_score_ids, bs * len, len, len, 0);

    gather_attention_rows(Q, middle_rows, middle_rows_ids, h, bs * middle_size);
    gather_attention_rows(K, global_cols, dense_row_ids, h,
                          bs * global_col_size);
    gather_attention_rows(V, global_cols_v, dense_row_ids, h,
                          bs * global_col_size);
    gather_attention_rows(K, rand_cols, rand_cols_ids, h,
                          bs * rand_col_size * middle_block_num);
    gather_attention_rows(V, rand_cols_v, rand_cols_ids, h,
                          bs * rand_col_size * middle_block_num);

    // WE DO NOT GATHER WINDOW DATA.
    cublasSgemmStridedBatchedQK(handle, middle_rows, global_cols, global_score,
                                middle_size, global_col_size, h, bs);
    cublasSgemmStridedBatchedQK(handle, middle_rows, rand_cols, rand_score,
                                blksz, rand_col_size, h, bs * middle_block_num);
    // compute window score
    cublasSgemmStridedBatchedQWindowK(handle, Q + gs * blksz * h, K,
                                      window_score, blksz, ws * blksz, h, blksz,
                                      bs * blk_num);

    // scatter global/rand score[b, middlesize, gs/rs] to
    // middle_score_tmp[b, 2blksz:len-2*blksz, 0:/gs:]
    scatter_attention_rows_with_stride(
        global_score, middle_score_tmp, scatter_score_ids, gs * blksz,
        (gs + ws + rs) * blksz, bs * middle_size);
    scatter_attention_rows_with_stride(
        rand_score, middle_score_tmp + gs * blksz, scatter_score_ids,
        gs * blksz, (gs + ws + rs) * blksz, bs * middle_size);
    scatter_attention_rows_with_stride(
        window_score, middle_score_tmp + (gs + rs) * blksz,
        window_scatter_score_ids, ws * blksz, (gs + ws + rs) * blksz, bs * len);

    CublasCheck(cublasSscal_v2(handle, bs * len * (gs + rs + ws) * blksz, &scal,
                               middle_score_tmp, 1));
    attention_score_softmax_op(middle_score_tmp, middle_score,
                               (gs + rs + ws) * blksz, bs * len);

    gather_attention_rows_with_stride(middle_score, global_score,
                                      scatter_score_ids, gs * blksz,
                                      (gs + ws + rs) * blksz, middle_size * bs);
    gather_attention_rows_with_stride(middle_score + gs * blksz, rand_score,
                                      scatter_score_ids, rs * blksz,
                                      (gs + ws + rs) * blksz, middle_size * bs);
    gather_attention_rows_with_stride(
        middle_score + gs * blksz + rs * blksz, window_score,
        window_scatter_score_ids, ws * blksz, (gs + ws + rs) * blksz, bs * len);
    cublasSgemmStridedBatchedSV(handle, global_score, global_cols_v, global_res,
                                middle_size, h, gs * blksz, bs);
    cublasSgemmStridedBatchedSV(handle, rand_score, rand_cols_v, rand_res, 1, h,
                                rand_col_size, bs * middle_block_num);
    cublasSgemmStridedBatchedSWindowV(handle, window_score, V, window_res,
                                      blksz, h, ws * blksz, blksz,
                                      bs * blk_num);
    // reduce sum
    CublasCheck(cublasSaxpy_v2(handle, bs * middle_size * h, &alpha, global_res,
                               1, rand_res, 1));
    gather_attention_rows(window_res, global_res, scatter_score_ids, h,
                          middle_size * bs);
    CublasCheck(cublasSaxpy_v2(handle, bs * middle_size * h, &alpha, global_res,
                               1, rand_res, 1));

    scatter_attention_rows(rand_res, Res, middle_rows_ids, h, bs * middle_size);
}

void SpecialRows(cublasHandle_t& handle, const float* Q, const float* K,
                 const float* V, float* Res, int64_t* special_row_ids,
                 int64_t* special_col_ids, float* special_row_pair,
                 float* special_K_cols, float* special_V_cols, int col_num,
                 float* special_score_tmp, float* special_score,
                 float* special_res, float scal, int block_row_idx,
                 int64_t window_size, int64_t h, int64_t bs, int64_t len,
                 int64_t blksz, int64_t blk_num, int64_t gs, int64_t ws,
                 int64_t rs) {
    build_special_row_ids(special_row_ids, block_row_idx, bs, len, blk_num,
                          blksz, gs, ws);
    // [TODO]: Did not write the correct index,
    // but the time consumption here is negligible.
    build_special_cols_ids(special_col_ids, block_row_idx, window_size, bs, len,
                           blk_num, blksz, gs, ws, rs);

    gather_attention_rows(Q, special_row_pair, special_row_ids, h,
                          bs * 2 * blksz);
    gather_attention_rows(K, special_K_cols, special_col_ids, h,
                          bs * 2 * col_num * blksz);
    gather_attention_rows(V, special_V_cols, special_col_ids, h,
                          bs * 2 * col_num * blksz);

    cublasSgemmStridedBatchedQK(handle, special_row_pair, special_K_cols,
                                special_score_tmp, blksz, col_num * blksz, h,
                                2 * bs);
    // S / d^(1/2)  -> S
    CublasCheck(cublasSscal_v2(handle, bs * 2 * blksz * col_num * blksz, &scal,
                               special_score_tmp, 1));

    attention_score_softmax_op(special_score_tmp, special_score,
                               col_num * blksz, bs * 2 * blksz);

    cublasSgemmStridedBatchedSV(handle, special_score, special_V_cols,
                                special_res, blksz, h, col_num * blksz, 2 * bs);

    scatter_attention_rows(special_res, Res, special_row_ids, h, 2 * blksz);
}

float bigbird(const float* Q, const float* K, const float* V, float* Res,
              int64_t len, int64_t h, int64_t blksz, int64_t bs,
              int64_t blk_num, int64_t gs, int64_t ws_, int64_t rs,
              std::vector<int64_t>& rand_attn_pos, int warmup = 20,
              int repeat = 100) {
    int ws = ws_ * 2 + 1;
    float scal = float(1.0 / std::sqrt(len));
    float alpha = 1.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    float* dense_row;
    float* dense_score;
    float* dense_score_tmp;
    float* dense_result;

    // First&last global block rows concat in blksz
    // dim to compute first&last block score in one BMM.
    // Actually we can also write dense row part result to this tensor.
    int64_t dense_row_size = gs * 2 * blksz;

    //[b, 2*gs*blksz, h]
    int64_t size = bs * dense_row_size * h * sizeof(float);
    CudaCheck(cudaMalloc(&dense_row, size));

    //[b, 2*gs*blksz, len]
    size = bs * dense_row_size * len * sizeof(float);
    CudaCheck(cudaMalloc(&dense_score, size));

    size = bs * dense_row_size * len * sizeof(float);
    CudaCheck(cudaMalloc(&dense_score_tmp, size));

    size = bs * dense_row_size * h * sizeof(float);
    CudaCheck(cudaMalloc(&dense_result, size));

    // Dense row index in input Q
    int64_t* dense_row_ids;
    size = bs * (gs * 2 * blksz) * sizeof(int64_t);
    // [b, 2*gs*blksz]
    CudaCheck(cudaMalloc(&dense_row_ids, size));

    /* -------------------------DENSE ROW PART--------------------------*/
    for (int i = 0; i < warmup; ++i) {
        DenseRows(handle, Q, K, V, dense_row_ids, dense_row, dense_score,
                  dense_score_tmp, dense_result, len, h, bs, gs, blksz,
                  dense_row_size, scal);
    }

    CudaTimer timer;
    timer.Start();
    for (int i = 0; i < repeat; ++i) {
        DenseRows(handle, Q, K, V, dense_row_ids, dense_row, dense_score,
                  dense_score_tmp, dense_result, len, h, bs, gs, blksz,
                  dense_row_size, scal);
    }
    float time_dense = timer.Stop() / repeat;

    // Store the result to where they should be.
    // just reuse the dense_row_ids for scatter.
    CudaCheck(cudaFree(dense_row));
    CudaCheck(cudaFree(dense_score));
    CudaCheck(cudaFree(dense_score_tmp));
    CudaCheck(cudaFree(dense_result));
    // will be used in MIDDLE ROW PART to select global KV cols.
    CudaCheck(cudaFree(dense_row_ids));

    /* -------------------------MIDDLE ROW PART------------------------*/
    int middle_block_num = blk_num - 2 * (gs + ws);
    int middle_size = middle_block_num * blksz;
    int global_col_size = 2 * gs * blksz;
    int rand_col_size = rs * blksz;
    int window_col_size = ws * blksz;

    float* middle_rows;
    float* global_cols;
    float* rand_cols;
    float* global_cols_v;
    float* rand_cols_v;
    float* middle_score;
    float* middle_score_tmp;
    float* global_score;
    float* rand_score;
    float* window_score;
    float* global_res;
    float* window_res;
    float* rand_res;

    size = bs * middle_size * h * sizeof(float);
    CudaCheck(cudaMalloc(&middle_rows, size));
    size = bs * global_col_size * h * sizeof(float);
    CudaCheck(cudaMalloc(&global_cols, size));
    size = bs * rand_col_size * middle_block_num * h * sizeof(float);
    CudaCheck(cudaMalloc(&rand_cols, size));
    size = bs * global_col_size * h * sizeof(float);
    CudaCheck(cudaMalloc(&global_cols_v, size));
    size = bs * rand_col_size * middle_block_num * h * sizeof(float);
    CudaCheck(cudaMalloc(&rand_cols_v, size));

    size = bs * len * (global_col_size + rand_col_size + window_col_size) *
           sizeof(float);
    CudaCheck(cudaMalloc(&middle_score, size));

    size = bs * len * (global_col_size + rand_col_size + window_col_size) *
           sizeof(float);
    CudaCheck(cudaMalloc(&middle_score_tmp, size));
    size = bs * global_col_size * middle_size * sizeof(float);
    CudaCheck(cudaMalloc(&global_score, size));
    size = bs * rand_col_size * middle_size * sizeof(float);
    CudaCheck(cudaMalloc(&rand_score, size));

    size = bs * window_col_size * len * sizeof(float);
    CudaCheck(cudaMalloc(&window_score, size));
    size = bs * middle_size * h * sizeof(float);
    CudaCheck(cudaMalloc(&global_res, size));
    size = bs * middle_size * h * sizeof(float);
    CudaCheck(cudaMalloc(&rand_res, size));
    size = bs * (len)*h * sizeof(float);
    CudaCheck(cudaMalloc(&window_res, size));

    int64_t* middle_rows_ids;
    int64_t* rand_cols_ids;
    int64_t* scatter_score_ids;  // use for concat scores.
    int64_t* window_scatter_score_ids;
    int64_t* rand_attn;

    size = bs * middle_size * sizeof(int64_t);
    CudaCheck(cudaMalloc(&middle_rows_ids, size));
    size = bs * rand_col_size * middle_block_num * sizeof(int64_t);
    CudaCheck(cudaMalloc(&rand_cols_ids, size));
    size = bs * middle_size * sizeof(int64_t);
    CudaCheck(cudaMalloc(&scatter_score_ids, size));
    size = middle_size * rs * sizeof(int64_t);
    CudaCheck(cudaMalloc(&rand_attn, size));
    size = bs * middle_size * sizeof(int64_t);
    CudaCheck(cudaMalloc(&window_scatter_score_ids, size));
    size = middle_size * rs * sizeof(int64_t);
    CudaCheck(cudaMemcpy(rand_attn, rand_attn_pos.data(), size,
                         cudaMemcpyHostToDevice));

    for (int i = 0; i > warmup; ++i) {
        MiddleRows(handle, Q, K, V, Res, middle_rows, global_cols,
                   global_cols_v, rand_cols, rand_cols_v, global_score,
                   rand_score, middle_score, window_score, middle_score_tmp,
                   global_res, window_res, rand_res, alpha, scal, dense_row_ids,
                   middle_rows_ids, rand_cols_ids, rand_attn, scatter_score_ids,
                   window_scatter_score_ids, middle_size, global_col_size,
                   rand_col_size, window_col_size, middle_block_num, h, bs, len,
                   blk_num, blksz, gs, ws, rs);
    }

    timer.Start();
    for (int i = 0; i > repeat; ++i) {
        MiddleRows(handle, Q, K, V, Res, middle_rows, global_cols,
                   global_cols_v, rand_cols, rand_cols_v, global_score,
                   rand_score, middle_score, window_score, middle_score_tmp,
                   global_res, window_res, rand_res, alpha, scal, dense_row_ids,
                   middle_rows_ids, rand_cols_ids, rand_attn, scatter_score_ids,
                   window_scatter_score_ids, middle_size, global_col_size,
                   rand_col_size, window_col_size, middle_block_num, h, bs, len,
                   blk_num, blksz, gs, ws, rs);
    }
    float time_window = timer.Stop() / repeat;

    CudaCheck(cudaFree(dense_row_ids));
    CudaCheck(cudaFree(middle_rows));
    CudaCheck(cudaFree(global_cols));
    CudaCheck(cudaFree(rand_cols));
    CudaCheck(cudaFree(global_cols_v));
    CudaCheck(cudaFree(rand_cols_v));
    CudaCheck(cudaFree(middle_score));
    CudaCheck(cudaFree(middle_score_tmp));
    CudaCheck(cudaFree(global_score));
    CudaCheck(cudaFree(rand_score));
    CudaCheck(cudaFree(window_score));
    CudaCheck(cudaFree(global_res));
    CudaCheck(cudaFree(window_res));
    CudaCheck(cudaFree(rand_res));

    /* -----------------------SPECIAL ROW PART--------------------------*/
    float time_special = 0.;
    for (int block_row_idx = gs; block_row_idx < gs + ws; ++block_row_idx) {
        int window_size = block_row_idx - gs + ws + 1;
        int col_num = gs + rs + window_size;
        float* special_row_pair;
        float* special_K_cols;
        float* special_V_cols;
        float* special_score;
        float* special_score_tmp;
        float* special_res;

        size = bs * 2 * col_num * blksz * h * sizeof(float);
        CudaCheck(cudaMalloc(&special_K_cols, size));

        size = bs * 2 * col_num * blksz * h * sizeof(float);
        CudaCheck(cudaMalloc(&special_V_cols, size));

        size = bs * 2 * blksz * h * sizeof(float);
        CudaCheck(cudaMalloc(&special_row_pair, size));

        size = bs * 2 * blksz * col_num * blksz * sizeof(float);
        CudaCheck(cudaMalloc(&special_score, size));

        size = bs * 2 * blksz * col_num * blksz * sizeof(float);
        CudaCheck(cudaMalloc(&special_score_tmp, size));

        size = bs * 2 * blksz * h * sizeof(float);
        CudaCheck(cudaMalloc(&special_res, size));

        int64_t* special_row_ids;
        int64_t* special_col_ids;
        size = bs * 2 * blksz * sizeof(int64_t);
        CudaCheck(cudaMalloc(&special_row_ids, size));
        size = bs * 2 * col_num * blksz * sizeof(int64_t);
        CudaCheck(cudaMalloc(&special_col_ids, size));

        for (int i = 0; i < warmup; ++i) {
            SpecialRows(handle, Q, K, V, Res, special_row_ids, special_col_ids,
                        special_row_pair, special_K_cols, special_V_cols,
                        col_num, special_score_tmp, special_score, special_res,
                        scal, block_row_idx, window_size, h, bs, len, blksz,
                        blk_num, gs, ws, rs);
        }

        timer.Start();
        for (int i = 0; i < repeat; ++i) {
            SpecialRows(handle, Q, K, V, Res, special_row_ids, special_col_ids,
                        special_row_pair, special_K_cols, special_V_cols,
                        col_num, special_score_tmp, special_score, special_res,
                        scal, block_row_idx, window_size, h, bs, len, blksz,
                        blk_num, gs, ws, rs);
        }
        time_special += (timer.Stop() / repeat);

        CudaCheck(cudaFree(special_col_ids));
        CudaCheck(cudaFree(special_row_ids));
        CudaCheck(cudaFree(special_row_pair));
        CudaCheck(cudaFree(special_K_cols));
        CudaCheck(cudaFree(special_V_cols));
        CudaCheck(cudaFree(special_score));
        CudaCheck(cudaFree(special_score_tmp));
        CudaCheck(cudaFree(special_res));
    }

    CublasCheck(cublasDestroy(handle));

    return time_dense + time_window + time_special;
}

void run_test(int len) {
    const int gs = 1;  // global size
    const int ws = 1;  // [i-ws, ..., i, ..., i+ws] //
    const int rs = 1;  // random size

    const int bs = 32;
    const int h = 512;     //>=32
    const int blksz = 64;  // block size

    int64_t blk_num = len / blksz;

    // generate random attended positions.
    std::vector<int64_t> rand_attn_pos;
    init_rand_attn(len, blksz, rs, ws, gs, rand_attn_pos);

    // init QKV and Res on GPU.
    float* Q;
    float* K;
    float* V;
    float* Res;

    int64_t data_size = bs * len * h * sizeof(float);

    // add tails memory for strided BMM
    CudaCheck(cudaMalloc(&Q, data_size + 2 * blksz * h));
    CudaCheck(cudaMalloc(&K, data_size));
    CudaCheck(cudaMalloc(&V, data_size));
    CudaCheck(cudaMalloc(&Res, data_size));
    init_data(Q, K, V, Res, bs, len, h);

    float time = bigbird(Q, K, V, Res, len, h, blksz, bs, blk_num, gs, ws, rs,
                         rand_attn_pos);

    std::cout << bs << "\t" << len << "\t" << h << "\t" << blksz << "\t" << time
              << std::endl;

    CudaCheck(cudaFree(Q));
    CudaCheck(cudaFree(K));
    CudaCheck(cudaFree(V));
    CudaCheck(cudaFree(Res));
}
}  // namespace

int main(int argc, char* argv[]) {
    std::cout
        << "batch size\t sequence length\thidden\tblock size\telapsed time(ms)"
        << std::endl;

    run_test(4096);
    run_test(8192);
    return 0;
}
