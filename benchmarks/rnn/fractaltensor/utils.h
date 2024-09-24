#pragma once

#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/elementwise.h"
#include "kaleido/core/device/kernels/fill.h"
#include "kaleido/core/device/kernels/gather_scatter.h"
#include "kaleido/core/device/kernels/math_functor.h"
#include "kaleido/core/operators/concat_op.h"
#include "kaleido/core/operators/launch_config.h"
#include "kaleido/core/tensor.h"

#include <curand.h>
#include <glog/logging.h>

#include <iomanip>
#include <iostream>
#include <mutex>

#define CHECK_ERROR(call)                                                      \
    do {                                                                       \
        int _error = (call);                                                   \
        if (_error) {                                                          \
            printf("*** Error *** at [%s:%d] error=%d \n", __FILE__, __LINE__, \
                   _error);                                                    \
        }                                                                      \
    } while (0)

#define FUSE_ALL_ELEMENTWISE

std::once_flag glog_init_flag;
void InitGLOG(const std::string& prog_name) {
    std::call_once(glog_init_flag, [&]() {
        google::InitGoogleLogging(strdup(prog_name.c_str()));
    });
}

void printHeader() {
    std::cout << "ExpName\tShape\tGather(ms)\tCell-GEMM(ms)\t"
              << "Cell-Elementwise(ms)\tScatter(ms)\tTotal(ms)" << std::endl;
}

void PrintRecord(std::string shape, std::vector<std::string>& names,
                 std::vector<std::vector<float>>& times, int count,
                 bool print_header = true) {
    if (print_header) printHeader();

    std::cout << "FractalTensor\t" << shape;
    for (int i = 0; i < names.size(); ++i) {
        auto time = times[i];
        auto name = names[i];
        float total = accumulate(time.begin(), time.end(), 0.);

        if (i != names.size() - 1) continue;

        if (total > 0) {
            std::cout.setf(std::ios::fixed);
            std::cout << std::setprecision(3) << time[0] / count << "\t"
                      << time[1] / count << "\t" << time[2] / count << "\t"
                      << time[3] / count << "\t" << total / count << std::endl;
        }
    }
}

namespace kaleido {
namespace core {

// elapsed time in millisecond
#define CpuElapse(base, start) \
    base += ((double)(clock() - start)) * 1000 / CLOCKS_PER_SEC;

#define GpuElapse(start, stop, elapsed, total)   \
    cudaEventRecord(stop, 0);                    \
    cudaEventSynchronize(stop);                  \
    cudaEventElapsedTime(&elapsed, start, stop); \
    total += elapsed;

void RandomTensor(Tensor& input, float mean = 0, float stddev = 0.1) {
    float* data = input.mutable_data<float>();
    int num = static_cast<int>(input.numel());
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    curandGenerateNormal(prng, data, num, mean, stddev);
}

float FillZeros(Tensor& t, float val = 0.) {
    float elapsed = 0.;
    float total = 0.;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int block = 512;
    int n = t.numel();
    int grid = (n + block - 1) / block;

    cudaEventRecord(start, 0);
    cuda_kernel::KeFillValue<float>
        <<<grid, block>>>(t.mutable_data<float>(), t.numel(), val);

    GpuElapse(start, stop, elapsed, total);
    return total;
}

float GemmBatched(const GPUContext& context, const Tensor& x, const Tensor& w,
                  const Tensor& h, const Tensor& u, Tensor& tmp1,
                  Tensor& tmp2) {
    // GemmBatched is almost as faster as call GEMM multiple times.
    cublasHandle_t handle = context.cublas_handle();

    int m = x.dim_size(0);
    int n = w.dim_size(1);
    int k = x.dim_size(1);

    int lda = x.dim_size(1);
    int ldb = w.dim_size(1);
    int ldc = tmp1.dim_size(1);

    float alf = 1.;
    float bet = 0.;
    const float* alpha = &alf;
    const float* beta = &bet;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    const int batch_count = 2;
    const float* dAarrayTmp[batch_count];
    const float* dBarrayTmp[batch_count];
    float* dCarrayTmp[batch_count];

    dAarrayTmp[0] = x.data<float>();
    dAarrayTmp[1] = h.data<float>();

    dBarrayTmp[0] = w.data<float>();
    dBarrayTmp[1] = u.data<float>();

    dCarrayTmp[0] = tmp1.mutable_data<float>();
    dCarrayTmp[1] = tmp2.mutable_data<float>();

    float **devAarray, **devBarray, **devCarray;
    CudaCheck(cudaMalloc(&devAarray, batch_count * sizeof(float*)));
    CudaCheck(cudaMemcpy(devAarray, dAarrayTmp, batch_count * sizeof(float*),
                         cudaMemcpyHostToDevice));
    CudaCheck(cudaMalloc(&devBarray, batch_count * sizeof(float*)));
    CudaCheck(cudaMemcpy(devBarray, dBarrayTmp, batch_count * sizeof(float*),
                         cudaMemcpyHostToDevice));
    CudaCheck(cudaMalloc(&devCarray, batch_count * sizeof(float*)));
    CudaCheck(cudaMemcpy(devCarray, dCarrayTmp, batch_count * sizeof(float*),
                         cudaMemcpyHostToDevice));

    float total = 0.;
    float elapsed = 0.;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    CublasCheck(cublasSgemmBatched(handle, transb, transa, n, m, k, alpha,
                                   (const float**)devBarray, ldb,
                                   (const float**)devAarray, lda, beta,
                                   (float**)devCarray, ldc, batch_count));

    GpuElapse(start, stop, elapsed, total);

    cudaFree(devAarray);
    cudaFree(devBarray);
    cudaFree(devCarray);

    return total;
}

namespace {
void fillRandom(float* A, int elementNum) {
    // create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    // fill the array with random numbers on the device
    curandGenerateUniform(prng, A, elementNum);
}
}  // namespace

float Gemm(const GPUContext& context, const Tensor& a, const Tensor& b,
           Tensor& c) {
    // cublasHandle_t handle = context.cublas_handle();
    int m = a.dim_size(0);
    int k = a.dim_size(1);
    int n = b.dim_size(1);

    // NOTE: interprete the matrix as row-major.
    int lda = a.dim_size(1);
    int ldb = b.dim_size(1);
    int ldc = c.dim_size(1);

    int64_t size_A = m * k;
    int64_t size_B = k * n;
    int64_t size_C = m * n;

    // use the locally warmup cublas GEMM, and locally allocated GPU memory
    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;

    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    fillRandom(d_A, size_A);
    fillRandom(d_B, size_B);

    cublasHandle_t handle;
    cublasCreate(&handle);
    //========================

    float alf = 1.;
    float bet = 0.;
    const float* alpha = &alf;
    const float* beta = &bet;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    float total = 0.;
    float elapsed = 0.;

    for (int i = 0; i < 5; ++i)
        CublasCheck(cublasSgemm(handle, transb, transa, n, m, k, alpha, d_B,
                                ldb, d_A, lda, beta, d_C, ldc));

    int iter = 10;
    CudaTimer timer;
    timer.Start();
    for (int i = 0; i < 10; ++i) {
        CublasCheck(cublasSgemm(handle, transb, transa, n, m, k, alpha, d_B,
                                ldb, d_A, lda, beta, d_C, ldc));
    }
    float time = timer.Stop() / iter;

    // CublasCheck(cublasSgemm(handle, transb, transa, n, m, k, alpha,
    //                         b.data<float>(), ldb, a.data<float>(), lda, beta,
    //                         c.mutable_data<float>(), ldc));

    // ==== use locally allocated memory =====
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // ========================================
    return time;
}

float BMM(const GPUContext& context, const Tensor& a, const Tensor& b,
          Tensor& c, long long int stride_a, long long int stride_b,
          long long int stride_c, int m, int n, int k, int batch_count) {
    cublasHandle_t handle = context.cublas_handle();

    // NOTE: interprete the matrix as row-major.
    int lda = a.dim_size(-1);
    int ldb = b.dim_size(-1);
    int ldc = c.dim_size(-1);

    float alf = 1.;
    float bet = 0.;
    const float* alpha = &alf;
    const float* beta = &bet;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    float elapsed = 0.f;
    float total = 0.f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    CublasCheck(cublasSgemmStridedBatched(
        handle, transb, transa, n, m, k, alpha, b.data<float>(), ldb, stride_b,
        a.data<float>(), lda, stride_a, beta, c.mutable_data<float>(), ldc,
        stride_c, batch_count));

    GpuElapse(start, stop, elapsed, total);
    return total;
}

float FusedBMM(const GPUContext& context, const Tensor& a1, const Tensor& a2,
               const Tensor& b1, const Tensor& b2, Tensor& c1, Tensor& c2,
               long long int stride_a, long long int stride_b,
               long long int stride_c, int m, int n, int k, int batch_count) {
    /*
     This function computes y1 = x @ W and y2 = h @ U simutaneously in a
     single batched matrix multiplication.

     a1 stands for x, a2 stands for h;
     b1 stands for W, b2 stands for U;
     c1 stands for the result of x @ W, c2 stands for the result of h @
     U;
     */
    cublasHandle_t handle = context.cublas_handle();

    const int batch_count_new = batch_count * 2;
    const float* dAarrayTmp[batch_count_new];
    const float* dBarrayTmp[batch_count_new];
    float* dCarrayTmp[batch_count_new];

    for (int i = 0; i < batch_count; ++i) {
        // Batch Add the pointer to the array.
        dAarrayTmp[i] = a1.data<float>() + i * stride_a;
        dBarrayTmp[i] = b1.data<float>() + i * stride_b;
        dCarrayTmp[i] = c1.mutable_data<float>() + i * stride_c;
    }
    for (int i = 0; i < batch_count; ++i) {
        // Batch Add the pointer to the array.
        dAarrayTmp[batch_count + i] = a2.data<float>() + i * stride_a;
        dBarrayTmp[batch_count + i] = b2.data<float>() + i * stride_b;
        dCarrayTmp[batch_count + i] = c2.mutable_data<float>() + i * stride_c;
    }

    float **devAarray, **devBarray, **devCarray;
    CudaCheck(cudaMalloc(&devAarray, batch_count_new * sizeof(float*)));
    CudaCheck(cudaMemcpy(devAarray, dAarrayTmp,
                         batch_count_new * sizeof(float*),
                         cudaMemcpyHostToDevice));
    CudaCheck(cudaMalloc(&devBarray, batch_count_new * sizeof(float*)));
    CudaCheck(cudaMemcpy(devBarray, dBarrayTmp,
                         batch_count_new * sizeof(float*),
                         cudaMemcpyHostToDevice));
    CudaCheck(cudaMalloc(&devCarray, batch_count_new * sizeof(float*)));
    CudaCheck(cudaMemcpy(devCarray, dCarrayTmp,
                         batch_count_new * sizeof(float*),
                         cudaMemcpyHostToDevice));

    // NOTE: interprete the matrix as row-major.
    int lda = a1.dim_size(-1);
    int ldb = b1.dim_size(-1);
    int ldc = c1.dim_size(-1);

    float alf = 1.;
    float bet = 0.;
    const float* alpha = &alf;
    const float* beta = &bet;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    float elapsed = 0.f;
    float total = 0.f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    CublasCheck(cublasSgemmBatched(handle, transb, transa, n, m, k, alpha,
                                   (const float**)devBarray, ldb,
                                   (const float**)devAarray, lda, beta,
                                   (float**)devCarray, ldc, batch_count_new));

    GpuElapse(start, stop, elapsed, total);
    return total;
}

float Elementwise(const GPUContext& context, const Tensor& tmp1,
                  const Tensor& tmp2, const Tensor& c, Tensor& tmp3,
                  Tensor& tmp4, Tensor& out1, Tensor& out2) {
    float total = 0.;
    float elapsed = 0.;

    int threads;
    int blocks;
    ops::GetGpuLaunchConfig1D(context, tmp1.numel(), &threads, &blocks);
    dim3 block = dim3(threads, 1, 1);
    dim3 grid = dim3(blocks, 1, 1);

    // int block = 1024;
    // int grid = (tmp1.numel() + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // tmp3 = tmp2 + tmp1
    cuda_kernel::Add<float> add;
    cuda_kernel::ElementwiseBinaryKernel<float, cuda_kernel::Add<float>>
        <<<grid, block, 0>>>(tmp1.data<float>(), tmp2.data<float>(),
                             tmp3.mutable_data<float>(), tmp3.numel(), add);
    GpuElapse(start, stop, elapsed, total);

    // tmp4[0:3] = sigmoid(tmp3[0:3])
    const int64_t offset = tmp3.numel() / 4;
    ops::GetGpuLaunchConfig1D(context, offset * 3, &threads, &blocks);
    block = dim3(threads, 1, 1);
    grid = dim3(blocks, 1, 1);
    cuda_kernel::Sigmoid<float> sigmoid;
    cuda_kernel::ElementwiseUnaryKernel<float, cuda_kernel::Sigmoid<float>>
        <<<grid, block, 0>>>(tmp3.data<float>(), tmp4.mutable_data<float>(),
                             offset * 3, sigmoid);

    // tmp4[3:] = tanh(tmp3[3:])
    ops::GetGpuLaunchConfig1D(context, offset, &threads, &blocks);
    block = dim3(threads, 1, 1);
    grid = dim3(blocks, 1, 1);
    cuda_kernel::Tanh<float> tanh;
    cuda_kernel::ElementwiseUnaryKernel<float, cuda_kernel::Tanh<float>>
        <<<grid, block, 0>>>(tmp3.data<float>() + offset * 3,
                             tmp4.mutable_data<float>() + offset * 3, offset,
                             tanh);

    cuda_kernel::CellFunc<float> func1;
    cuda_kernel::ElementwiseArityFourKernel<float, cuda_kernel::CellFunc<float>>
        <<<grid, block, 0>>>(tmp4.data<float>() + offset /*f_{t}*/,
                             c.data<float>() /*c_{t-1}*/,
                             tmp4.data<float>() /*i_{t}*/,
                             tmp4.data<float>() + offset * 3 /*c_t*/,
                             out1.mutable_data<float>(), offset, func1);

    cuda_kernel::HiddenFunc<float> func2;
    cuda_kernel::ElementwiseBinaryKernel<float, cuda_kernel::HiddenFunc<float>>
        <<<grid, block, 0>>>(tmp4.data<float>() + 2 * offset /*o_{t}*/,
                             out1.data<float>() /*c_{t}*/,
                             out2.mutable_data<float>(), offset, func2);

    return total;
}

float ElementwiseFused(const GPUContext& context, const Tensor& tmp1,
                       const Tensor& tmp2, const Tensor& c, Tensor& out1,
                       Tensor& out2) {
    int64_t hidden = tmp1.dim_size(-1) / 4;
    int64_t remain_dim = tmp1.numel() / tmp1.dim_size(-1);

    Tensor igx({remain_dim, hidden}, nullptr);
    igx.CreateFrom<float>(tmp1, 0);

    Tensor fgx({remain_dim, hidden}, nullptr);
    fgx.CreateFrom<float>(tmp1, fgx.numel());

    Tensor ogx({remain_dim, hidden}, nullptr);
    ogx.CreateFrom<float>(tmp1, ogx.numel() * 2);

    Tensor cgx({remain_dim, hidden}, nullptr);
    cgx.CreateFrom<float>(tmp1, cgx.numel() * 3);

    // igu, fgu, ogu, cgu, should have a shape of [batch, hidden]
    Tensor igu({remain_dim, hidden}, nullptr);
    igu.CreateFrom<float>(tmp2, 0);

    Tensor fgu({remain_dim, hidden}, nullptr);
    fgu.CreateFrom<float>(tmp2, fgu.numel());

    Tensor ogu({remain_dim, hidden}, nullptr);
    ogu.CreateFrom<float>(tmp2, ogu.numel() * 2);

    Tensor cgu({remain_dim, hidden}, nullptr);
    cgu.CreateFrom<float>(tmp2, cgu.numel() * 3);

    float total = 0.;
    float elapsed = 0.;

    int threads;
    int blocks;
    ops::GetGpuLaunchConfig1D(context, igx.numel(), &threads, &blocks);
    dim3 block = dim3(threads, 1, 1);
    dim3 grid = dim3(blocks, 1, 1);

    // int block = 1024;
    // int grid = (igx.numel() + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cuda_kernel::ElementwiseLstmAct<float><<<grid, block, 0>>>(
        igx.data<float>(), igu.data<float>(), fgx.data<float>(),
        fgu.data<float>(), ogx.data<float>(), ogu.data<float>(),
        cgx.data<float>(), cgu.data<float>(), c.data<float>(),
        out1.mutable_data<float>(), out2.mutable_data<float>(), igx.numel());
    GpuElapse(start, stop, elapsed, total);

    return total;
}

std::vector<float> LstmCellBMM(const GPUContext& context, const Tensor& x,
                               const Tensor& w, const Tensor& c,
                               const Tensor& h, const Tensor& u, Tensor& out1,
                               Tensor& out2, Tensor& tmp1, Tensor& tmp2,
                               Tensor& tmp3, Tensor& tmp4, int m, int n, int k,
                               long long int stride_x, long long int stride_h,
                               int batch_count) {
    float t_gemm =
        BMM(context, x, w, tmp1, stride_x, w.dim_size(-1) * w.dim_size(-2),
            tmp1.dim_size(-1) * tmp1.dim_size(-2), m, n, k, batch_count);
    t_gemm +=
        BMM(context, h, u, tmp2, stride_h, u.dim_size(-1) * u.dim_size(-2),
            tmp2.dim_size(-1) * tmp2.dim_size(-2), m, n, k, batch_count);

#ifdef FUSE_ALL_ELEMENTWISE
    float t_elementwise = ElementwiseFused(context, tmp1, tmp2, c, out1, out2);
#else
    float t_elementwise =
        Elementwise(context, tmp1, tmp2, c, tmp3, tmp4, out1, out2);
#endif
    return {t_gemm, t_elementwise};
}

std::vector<float> LstmCellFuseBMM(const GPUContext& context, const Tensor& x,
                                   const Tensor& w, const Tensor& c,
                                   const Tensor& h, const Tensor& u,
                                   Tensor& out1, Tensor& out2, Tensor& tmp1,
                                   Tensor& tmp2, Tensor& tmp3, Tensor& tmp4,
                                   int m, int n, int k, long long int stride,
                                   int batch_count) {
    // stride_a: stride
    // stride_b: w.dim_size(-1) * w.dim_size(-2): 4 * hidden_size *
    // hidden_size stride_c: tmp1.dim_size(-1) * tmp1.dim_size(-2): kDepth3
    // * hidden_size stride_a: x, h stride_b: w, u stride_c: x @ w, h @ u
    float t_gemm =
        FusedBMM(context, x, h, w, u, tmp1, tmp2, stride,
                 w.dim_size(-1) * w.dim_size(-2),
                 tmp1.dim_size(-1) * tmp1.dim_size(-2), m, n, k, batch_count);

#ifdef FUSE_ALL_ELEMENTWISE
    float t_elementwise = ElementwiseFused(context, tmp1, tmp2, c, out1, out2);
#else
    float t_elementwise =
        Elementwise(context, tmp1, tmp2, c, tmp3, tmp4, out1, out2);
#endif
    return {t_gemm, t_elementwise};
}

std::vector<float> LstmCell(const GPUContext& context, const Tensor& x,
                            const Tensor& w, const Tensor& c, const Tensor& h,
                            const Tensor& u, Tensor& out1, Tensor& out2,
                            Tensor& tmp1, Tensor& tmp2, Tensor& tmp3,
                            Tensor& tmp4) {
    float t_gemm = GemmBatched(context, x, w, h, u, tmp1, tmp2);

#ifdef FUSE_ALL_ELEMENTWISE
    float t_elementwise = ElementwiseFused(context, tmp1, tmp2, c, out1, out2);
#else
    float t_elementwise =
        Elementwise(context, tmp1, tmp2, c, tmp3, tmp4, out1, out2);
#endif
    return {t_gemm, t_elementwise};
}

std::vector<float> LstmLayer(const GPUContext& context, const Tensor& xs,
                             const Tensor& w, const Tensor& c_init,
                             const Tensor h_init, const Tensor& u, Tensor& out1,
                             Tensor& out2, Tensor& tmp1, Tensor& tmp2,
                             Tensor& tmp3, Tensor& tmp4) {
    // input-2-hidden projection for the entire batch into a large GEMM.
    // W@xs

    float t_gemm = Gemm(context, xs, w, tmp1);

    Tensor tmp1_({out1.dim_size(1), w.dim_size(-1)}, nullptr);
    tmp1_.CreateFrom<float>(tmp1, 0);

    // hidden-2-hidden projection for time step 0.
    Tensor tmp2_({out1.dim_size(1), w.dim_size(-1)}, nullptr);
    tmp2_.CreateFrom<float>(tmp2, 0);
    // U@h_init
    t_gemm += Gemm(context, h_init, u, tmp2_);

    Tensor tmp3_({out1.dim_size(1), w.dim_size(-1)}, nullptr);
    Tensor tmp4_({out1.dim_size(1), w.dim_size(-1)}, nullptr);
    tmp3_.CreateFrom<float>(tmp3, 0);
    tmp4_.CreateFrom<float>(tmp4, 0);

    Tensor out1_({out1.dim_size(-2), out1.dim_size(-1)}, nullptr);
    out1_.CreateFrom<float>(out1, 0);

    Tensor out2_({out1.dim_size(-2), out1.dim_size(-1)}, nullptr);
    out2_.CreateFrom<float>(out2, 0);

    // LSTM element-wise activation for time step 0.
#ifdef FUSE_ALL_ELEMENTWISE
    float t_elementwise =
        ElementwiseFused(context, tmp1_, tmp2_, c_init, out1, out2);
#else
    float t_elementwise =
        Elementwise(context, tmp1_, tmp2_, c_init, tmp3_, tmp4_, out1_, out2_);
#endif

    Tensor c({out1.dim_size(-2), out1.dim_size(-1)}, nullptr);
    Tensor h({out2.dim_size(-2), out2.dim_size(-1)}, nullptr);

    for (size_t i = 1; i < out1.dim_size(0); ++i) {
        out1_.CreateFrom<float>(out1, i * out1_.numel());
        out2_.CreateFrom<float>(out2, i * out2_.numel());

        tmp1_.CreateFrom<float>(tmp1, i * tmp1_.numel());

        h.CreateFrom<float>(out1, (i - 1) * h.numel());
        c.CreateFrom<float>(out2, (i - 1) * c.numel());

        tmp2_.CreateFrom<float>(tmp2, i * tmp2_.numel());
        tmp3_.CreateFrom<float>(tmp3, i * tmp3_.numel());
        tmp4_.CreateFrom<float>(tmp4, i * tmp4_.numel());

        t_gemm += Gemm(context, h, u, tmp2_);

#ifdef FUSE_ALL_ELEMENTWISE
        t_elementwise += ElementwiseFused(context, tmp1_, tmp2_, c, out1, out2);
#else
        t_elementwise +=
            Elementwise(context, tmp1_, tmp2_, c, tmp3_, tmp4_, out1_, out2_);
#endif
    }

    return {t_gemm, t_elementwise};
}

float Elementwise_Vanilla(const GPUContext& context, const Tensor& tmp1,
                          const Tensor& tmp2, const Tensor& tmp3,
                          const Tensor& b, Tensor& tmp4, Tensor& out1,
                          Tensor& out2, int batch_count) {
    float total = 0.;
    float elapsed = 0.;

    int64_t num = tmp1.numel();
    int threads;
    int blocks;
    ops::GetGpuLaunchConfig1D(context, tmp1.numel(), &threads, &blocks);

    dim3 block_dims = dim3(threads, 1, 1);
    dim3 grid_dims = dim3(blocks, 1, 1);

    cuda_kernel::Add<float> add;
    cuda_kernel::AddAndTanh<float> addandtanh;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    if (batch_count == 1) return total;

    cuda_kernel::ElementwiseBinaryKernel<float, cuda_kernel::Add<float>>
        <<<grid_dims, block_dims, 0>>>(
            tmp3.data<float>() /*x_t@w*/, b.data<float>() /*state@u*/,
            tmp4.mutable_data<float>() /*h_x*/, num, add);
    cuda_kernel::ElementwiseBinaryKernel<float, cuda_kernel::AddAndTanh<float>>
        <<<grid_dims, block_dims, 0>>>(
            tmp1.data<float>() /*x_t@w*/, tmp4.data<float>() /*state@u*/,
            out1.mutable_data<float>() /*h_x*/, num, addandtanh);
    cuda_kernel::ElementwiseBinaryKernel<float, cuda_kernel::AddAndTanh<float>>
        <<<grid_dims, block_dims, 0>>>(
            tmp2.data<float>() /*x_t@w*/, tmp4.data<float>() /*state@u*/,
            out2.mutable_data<float>() /*h_x*/, num, addandtanh);

    GpuElapse(start, stop, elapsed, total);
    CHECK_ERROR(cudaGetLastError());

    return total;
}

std::vector<float> VanillaRNNCellBMM(const GPUContext& context,
                                     const Tensor& x_t, const Tensor& y_t,
                                     const Tensor& state, const Tensor& w,
                                     const Tensor& u, const Tensor& b,
                                     Tensor& tmp1, Tensor& tmp2, Tensor& tmp3,
                                     Tensor& tmp4, Tensor& h_x, Tensor& h_y,
                                     int m, int n, int k, int batch_count) {
    float t_gemm = 0.;
    float t_elementwise = 0.;

    t_gemm += BMM(context, x_t, w, tmp1, x_t.dim_size(-1) * x_t.dim_size(-2),
                  w.dim_size(-1) * w.dim_size(-2),
                  tmp1.dim_size(-1) * tmp1.dim_size(-2), m, n, k, batch_count);
    t_gemm += BMM(context, y_t, w, tmp2, y_t.dim_size(-1) * y_t.dim_size(-2),
                  w.dim_size(-1) * w.dim_size(-2),
                  tmp2.dim_size(-1) * tmp2.dim_size(-2), m, n, k, batch_count);
    CHECK_ERROR(cudaGetLastError());

    t_gemm +=
        BMM(context, state, u, tmp3, state.dim_size(-1) * state.dim_size(-2),
            u.dim_size(-1) * u.dim_size(-2),
            tmp3.dim_size(-1) * tmp3.dim_size(-2), m, n, 2 * k, batch_count);
    CHECK_ERROR(cudaGetLastError());

    t_elementwise += Elementwise_Vanilla(context, tmp1, tmp2, tmp3, b, tmp4,
                                         h_x, h_y, batch_count);
    CHECK_ERROR(cudaGetLastError());

    return {t_gemm, t_elementwise};
}

}  // namespace core
}  // namespace kaleido
