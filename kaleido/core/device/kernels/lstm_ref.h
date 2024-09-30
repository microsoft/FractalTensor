#pragma once
#include "kaleido/core/device/cuda_utils.h"

#include <math.h>

namespace kaleido::core::cuda_kernel {

template <typename T>
__global__ void MatrixAdd(const T* A, const T* B, T* C, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        C[i * n + j] = A[i * n + j] + B[i * n + j];
    }
}

template <typename T>
__global__ void Sigmod(T* A, int m, int n) {}

template <>
__global__ void Sigmod<__half>(__half* A, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        float xf = __half2float(A[i * n + j]);
        float sigmoid = 1.0f / (1.0f + expf(-xf));
        A[i * n + j] = __float2half(sigmoid);
    }
}

template <typename T>
__global__ void Tanh(T* A, int m, int n) {}

template <>
__global__ void Tanh<__half>(__half* A, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        float xf = __half2float(A[i * n + j]);
        float tanh = tanhf(xf);
        A[i * n + j] = __float2half(tanh);
    }
}

__global__ void HalfToFloatKernel(const __half* input, float* output,
                                  int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        output[idx] = static_cast<float>(input[idx]);
    }
}

__global__ void FloatToHalfKernel(const float* input, __half* output,
                                  int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        output[idx] = __float2half(input[idx]);
    }
}

template <typename T>
struct Cublas2GemmAdd {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k, const T* A,
                    int lda, const T* B, int ldb, const T* C, int ldc,
                    const T* D, T* E);
};

template <>
struct Cublas2GemmAdd<cutlass::half_t> {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const cutlass::half_t* A, int lda, const cutlass::half_t* B,
                    int ldb, const cutlass::half_t* C, int ldc,
                    const cutlass::half_t* D, cutlass::half_t* E) {
        const __half alpha = 1.0;
        const __half beta = 0.0;

        __half *acc1, *acc2;
        cudaMalloc((void**)&acc1, m * n * sizeof(__half));
        cudaMalloc((void**)&acc2, m * n * sizeof(__half));

        //
        // E = A@B + C@D
        CublasCheck(cublasHgemm(handle, transa, transb, m, n, k, &alpha,
                                reinterpret_cast<const __half*>(A), lda,
                                reinterpret_cast<const __half*>(B), ldb, &beta,
                                acc1, ldc));

        CublasCheck(cublasHgemm(handle, transa, transb, m, n, k, &alpha,
                                reinterpret_cast<const __half*>(C), lda,
                                reinterpret_cast<const __half*>(D), ldb, &beta,
                                acc2, ldc));

        cudaDeviceSynchronize();

        dim3 blockDim(32, 32);
        dim3 gridDim((m + 31) / 32, (n + 31) / 32);
        MatrixAdd<__half><<<gridDim, blockDim>>>(
            acc1, acc2, reinterpret_cast<__half*>(E), m, n);
        cudaDeviceSynchronize();

        cudaFree(acc1);
        cudaFree(acc2);
    }
};

// sigmod(A@B + C@D)
template <typename T>
struct Cublas2GemmAddSigmod {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k, const T* A,
                    int lda, const T* B, int ldb, const T* C, int ldc,
                    const T* D, T* E);
};

template <>
struct Cublas2GemmAddSigmod<cutlass::half_t> {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const cutlass::half_t* A, int lda, const cutlass::half_t* B,
                    int ldb, const cutlass::half_t* C, int ldc,
                    const cutlass::half_t* D, cutlass::half_t* E) {
        const __half alpha = 1.0;
        const __half beta = 0.0;

        __half *acc1, *acc2;
        cudaMalloc((void**)&acc1, m * n * sizeof(__half));
        cudaMalloc((void**)&acc2, m * n * sizeof(__half));

        //
        // E = A@B + C@D
        CublasCheck(cublasHgemm(handle, transa, transb, m, n, k, &alpha,
                                reinterpret_cast<const __half*>(A), lda,
                                reinterpret_cast<const __half*>(B), ldb, &beta,
                                acc1, ldc));

        CublasCheck(cublasHgemm(handle, transa, transb, m, n, k, &alpha,
                                reinterpret_cast<const __half*>(C), lda,
                                reinterpret_cast<const __half*>(D), ldb, &beta,
                                acc2, ldc));

        cudaDeviceSynchronize();

        dim3 blockDim(32, 32);
        dim3 gridDim((m + 31) / 32, (n + 31) / 32);

        MatrixAdd<__half><<<gridDim, blockDim>>>(
            acc1, acc2, reinterpret_cast<__half*>(E), m, n);
        cudaDeviceSynchronize();

        Sigmod<__half>
            <<<gridDim, blockDim>>>(reinterpret_cast<__half*>(E), m, n);
        cudaDeviceSynchronize();

        cudaFree(acc1);
        cudaFree(acc2);
    }
};

// tanh(A@B + C@D)
template <typename T>
struct Cublas2GemmAddTanh {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k, const T* A,
                    int lda, const T* B, int ldb, const T* C, int ldc,
                    const T* D, T* E);
};

template <>
struct Cublas2GemmAddTanh<cutlass::half_t> {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const cutlass::half_t* A, int lda, const cutlass::half_t* B,
                    int ldb, const cutlass::half_t* C, int ldc,
                    const cutlass::half_t* D, cutlass::half_t* E) {
        const __half alpha = 1.0;
        const __half beta = 0.0;

        __half *acc1, *acc2;
        cudaMalloc((void**)&acc1, m * n * sizeof(__half));
        cudaMalloc((void**)&acc2, m * n * sizeof(__half));

        //
        // E = A@B + C@D
        CublasCheck(cublasHgemm(handle, transa, transb, m, n, k, &alpha,
                                reinterpret_cast<const __half*>(A), lda,
                                reinterpret_cast<const __half*>(B), ldb, &beta,
                                acc1, ldc));

        CublasCheck(cublasHgemm(handle, transa, transb, m, n, k, &alpha,
                                reinterpret_cast<const __half*>(C), lda,
                                reinterpret_cast<const __half*>(D), ldb, &beta,
                                acc2, ldc));

        cudaDeviceSynchronize();

        dim3 blockDim(32, 32);
        dim3 gridDim((m + 31) / 32, (n + 31) / 32);

        MatrixAdd<__half><<<gridDim, blockDim>>>(
            acc1, acc2, reinterpret_cast<__half*>(E), m, n);
        cudaDeviceSynchronize();

        Tanh<__half><<<gridDim, blockDim>>>(reinterpret_cast<__half*>(E), m, n);
        cudaDeviceSynchronize();

        cudaFree(acc1);
        cudaFree(acc2);
    }
};

template <typename T>
struct CublasLSTMGate {
    void operator()(cublasHandle_t handle, int m, int n, int k, const T* W,
                    int lda, const T* x_t, int ldb, const T* U, int ldc,
                    const T* h_t_1, T* O);
};

template <>
struct CublasLSTMGate<cutlass::half_t> {
    void operator()(cublasHandle_t handle, int m, int n, int k,
                    const cutlass::half_t* W, const cutlass::half_t* x_t,
                    const cutlass::half_t* U, const cutlass::half_t* h_t_1,
                    cutlass::half_t* O) {
        const __half alpha = 1.0;
        const __half beta = 0.0;

        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;

        Cublas2GemmAddSigmod<cutlass::half_t> gemm_add_sigmod;
        Cublas2GemmAddTanh<cutlass::half_t> gemm_add_tanh;
        Cublas2GemmAdd<cutlass::half_t> gemm_add;

        /**
         * 1. i_t = sigmod(W_i@x_t + U_i@h_t_1)
         * 2. f_t = sigmod(W_f@x_t + U_f@h_t_1)
         * 3. o_t = sigmod(W_o@x_t + U_o@h_t_1)
         */
        gemm_add_sigmod(handle, transa, transb, n, 3 * m, k, x_t, k, W, k,
                        h_t_1, n, U, O);

        // 4. c_t_bar = tanh(W_c@x_t + U_c@h_t_1)
        gemm_add_tanh(handle, transa, transb, n, m, k, x_t, k, W + 3 * m * k, k,
                      h_t_1, n, U + 3 * m * k, O + 3 * m * n);
    }
};

template <typename T>
struct CpuLSTMElementWise {
    void operator()(const T* I_t, const T* F_t, const T* O_t, const T* C_t_bar,
                    const T* C_t_1, T* C_t, T* H_t);
};

template <>
struct CpuLSTMElementWise<float> {
    void operator()(const float* I_t, const float* F_t, const float* O_t,
                    const float* C_t_bar, const float* C_t_1, float* C_t,
                    float* H_t, int M, int N) {
        float *h_i_t, *h_f_t, *h_o_t, *h_c_t_bar, *h_c_t_1, *h_c_t, *h_h_t;

        h_i_t = (float*)malloc(M * N * sizeof(float));
        h_f_t = (float*)malloc(M * N * sizeof(float));
        h_o_t = (float*)malloc(M * N * sizeof(float));
        h_c_t_bar = (float*)malloc(M * N * sizeof(float));
        h_c_t_1 = (float*)malloc(M * N * sizeof(float));
        h_c_t = (float*)malloc(M * N * sizeof(float));
        h_h_t = (float*)malloc(M * N * sizeof(float));

        // Copy
        cudaMemcpy(h_i_t, I_t, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_f_t, F_t, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_o_t, O_t, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_t_bar, C_t_bar, M * N * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_t_1, C_t_1, M * N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // h_c_t = f_t * c_t_1 + i_t * c_t_bar
        for (int i = 0; i < M * N; i++) {
            h_c_t[i] = h_f_t[i] * h_c_t_1[i] + h_i_t[i] * h_c_t_bar[i];
        }

        // h_t = o_t * tanh(c_t)
        for (int i = 0; i < M * N; i++) {
            h_h_t[i] = h_o_t[i] * tanh(h_c_t[i]);
        }

        // Copy
        cudaMemcpy(C_t, h_c_t, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(H_t, h_h_t, M * N * sizeof(float), cudaMemcpyHostToDevice);
    }
};

template <>
struct CpuLSTMElementWise<cutlass::half_t> {
    void operator()(const cutlass::half_t* I_t, const cutlass::half_t* F_t,
                    const cutlass::half_t* O_t, const cutlass::half_t* C_t_bar,
                    const cutlass::half_t* C_t_1, cutlass::half_t* C_t,
                    cutlass::half_t* H_t, int M, int N) {
        float *d_I_t, *d_F_t, *d_O_t, *d_C_t_bar, *d_C_t_1, *d_C_t, *d_H_t;
        float *h_I_t, *h_F_t, *h_O_t, *h_C_t_bar, *h_C_t_1, *h_C_t, *h_H_t;

        // Allocate memory for d_I_t, d_F_t, d_O_t, d_C_t_bar, d_C_t, d_H_t
        cudaMalloc((void**)&d_I_t, M * N * sizeof(float));
        cudaMalloc((void**)&d_F_t, M * N * sizeof(float));
        cudaMalloc((void**)&d_O_t, M * N * sizeof(float));
        cudaMalloc((void**)&d_C_t_bar, M * N * sizeof(float));
        cudaMalloc((void**)&d_C_t_1, M * N * sizeof(float));
        cudaMalloc((void**)&d_C_t, M * N * sizeof(float));
        cudaMalloc((void**)&d_H_t, M * N * sizeof(float));

        // Allocate memory for h_I_t, h_F_t, h_O_t, h_C_t_bar, h_C_t, h_H_t
        h_I_t = (float*)malloc(M * N * sizeof(float));
        h_F_t = (float*)malloc(M * N * sizeof(float));
        h_O_t = (float*)malloc(M * N * sizeof(float));
        h_C_t_bar = (float*)malloc(M * N * sizeof(float));
        h_C_t_1 = (float*)malloc(M * N * sizeof(float));
        h_C_t = (float*)malloc(M * N * sizeof(float));
        h_H_t = (float*)malloc(M * N * sizeof(float));

        // __half to float
        HalfToFloatKernel<<<(M * N + 255) / 256, 256>>>(
            reinterpret_cast<const __half*>(I_t), d_I_t, M * N);
        HalfToFloatKernel<<<(M * N + 255) / 256, 256>>>(
            reinterpret_cast<const __half*>(F_t), d_F_t, M * N);
        HalfToFloatKernel<<<(M * N + 255) / 256, 256>>>(
            reinterpret_cast<const __half*>(O_t), d_O_t, M * N);
        HalfToFloatKernel<<<(M * N + 255) / 256, 256>>>(
            reinterpret_cast<const __half*>(C_t_bar), d_C_t_bar, M * N);
        HalfToFloatKernel<<<(M * N + 255) / 256, 256>>>(
            reinterpret_cast<const __half*>(C_t_1), d_C_t_1, M * N);

        // Copy Memory from Device to Host
        cudaMemcpy(h_I_t, d_I_t, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_F_t, d_F_t, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_O_t, d_O_t, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_t_bar, d_C_t_bar, M * N * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_t_1, d_C_t_1, M * N * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // Compute
        // h_c_t = f_t * h_c_t_1 + i_t * h_c_t_bar
        for (int i = 0; i < M * N; i++) {
            h_C_t[i] = h_F_t[i] * h_C_t_1[i] + h_I_t[i] * h_C_t_bar[i];
        }

        // h_t = o_t * tanh(c_t)
        for (int i = 0; i < M * N; i++) {
            h_H_t[i] = h_O_t[i] * tanh(h_C_t[i]);
        }

        // Copy Memory from Host to Device
        cudaMemcpy(d_C_t, h_C_t, M * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_H_t, h_H_t, M * N * sizeof(float), cudaMemcpyHostToDevice);

        // float to __half
        FloatToHalfKernel<<<(M * N + 255) / 256, 256>>>(
            d_C_t, reinterpret_cast<__half*>(C_t), M * N);
        FloatToHalfKernel<<<(M * N + 255) / 256, 256>>>(
            d_H_t, reinterpret_cast<__half*>(H_t), M * N);

        // Free Memory
        cudaFree(d_I_t);
        cudaFree(d_F_t);
        cudaFree(d_O_t);
        cudaFree(d_C_t_bar);
        cudaFree(d_C_t_1);
        cudaFree(d_C_t);
        cudaFree(d_H_t);
        free(h_I_t);
        free(h_F_t);
        free(h_O_t);
        free(h_C_t_bar);
        free(h_C_t_1);
        free(h_C_t);
        free(h_H_t);
    }
};

template <typename T>
struct ReferenceLSTMCell {
    void operator()(cublasHandle_t handle, T* w, const T* x, const T* u,
                    const T* c_1, const T* h_1, T* c, T* h, T* o, int m, int n,
                    int k);
};

template <>
struct ReferenceLSTMCell<cutlass::half_t> {
    void operator()(cublasHandle_t handle, const cutlass::half_t* w,
                    const cutlass::half_t* x, const cutlass::half_t* u,
                    const cutlass::half_t* c_1, const cutlass::half_t* h_1,
                    cutlass::half_t* c, cutlass::half_t* h, cutlass::half_t* o,
                    int m, int n, int k) {
        // cublasHandle_t handle;
        // CublasCheck(cublasCreate(&handle));

        cuda_kernel::CublasLSTMGate<cutlass::half_t> lstm_gate_reference;
        lstm_gate_reference(handle, m, n, k, w, x, u, h_1, o);

        // CublasCheck(cublasDestroy(handle));

        const cutlass::half_t* i_t = o;
        const cutlass::half_t* f_t = o + m * n;
        const cutlass::half_t* o_t = o + 2 * m * n;
        const cutlass::half_t* c_t_bar = o + 3 * m * n;

        cuda_kernel::CpuLSTMElementWise<cutlass::half_t>
            lstm_element_wise_reference;

        lstm_element_wise_reference(i_t, f_t, o_t, c_t_bar, c_1, c, h, m, n);
    }
};
}  // namespace kaleido::core::cuda_kernel
