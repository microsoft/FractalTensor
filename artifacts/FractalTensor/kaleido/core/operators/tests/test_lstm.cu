#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/cutlass_wmma.h"
#include "kaleido/core/device/kernels/gemm.h"
#include "kaleido/core/device/kernels/lstm.h"
#include "kaleido/core/operators/expect_eq_op.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/print_op.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tile_shape.h"

#include <cutlass/gemm/device/gemm.h>
#include <gtest/gtest.h>

namespace kaleido {
namespace core {

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_2gemm_add() {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    cudaDeviceProp m_dev_prop;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&m_dev_prop, device_idx);

    // GEMM shape
    int kM = dim_size<0, WholeShape>;
    int kN = dim_size<1, WholeShape>;
    int kK = dim_size<2, WholeShape>;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor A({kM, kK}, allocator);
    kaleido::core::Tensor B({kK, kN}, allocator);
    kaleido::core::Tensor C({kM, kK}, allocator);
    kaleido::core::Tensor D({kK, kN}, allocator);
    kaleido::core::Tensor E({kM, kN}, allocator);

    kaleido::core::Tensor acc1({kM, kN}, allocator);
    kaleido::core::Tensor acc2({kM, kN}, allocator);
    kaleido::core::Tensor ref_E({kM, kN}, allocator);

    fill(A, 0, 1e-3);
    fill(B, 0, 1e-3);
    fill(C, 0, 1e-3);
    fill(D, 0, 1e-3);
#ifdef DEBUG
    fill(A, "seq");
    fill(B, "seq");
#endif

    fill(E, 0.);
    fill(acc1, 0.);
    fill(acc2, 0.);
    fill(ref_E, 0.);

    using GemmAdd =
        cuda_kernel::Cute2GemmAdd<Element, InstructionShape, ValueMnk,
                                  WarpArrangement, CtaTileShape, WholeShape>;
    GemmAdd gemm_add;
    gemm_add(A.data<Element>(), B.data<Element>(), C.data<Element>(),
             D.data<Element>(), E.mutable_data<Element>());

    cuda_kernel::Cublas2GemmAdd<Element> gemm_add_reference;
    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));

    gemm_add_reference(handle, CUBLAS_OP_T, CUBLAS_OP_N, kN, kM, kK,
                       B.data<Element>(), B.dim_size(0), A.data<Element>(),
                       A.dim_size(1), D.data<Element>(), ref_E.dim_size(1),
                       C.data<Element>(), ref_E.mutable_data<Element>());

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(E, ref_E, 5e-3);

    std::cout << "Passed unittest for [" << kM << ", " << kN << ", " << kK
              << "]" << std::endl;
}

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_gemm_add_sigmod() {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    cudaDeviceProp m_dev_prop;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&m_dev_prop, device_idx);

    // GEMM shape
    int kM = dim_size<0, WholeShape>;
    int kN = dim_size<1, WholeShape>;
    int kK = dim_size<2, WholeShape>;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor A({kM, kK}, allocator);
    kaleido::core::Tensor B({kK, kN}, allocator);
    kaleido::core::Tensor C({kM, kK}, allocator);
    kaleido::core::Tensor D({kK, kN}, allocator);
    kaleido::core::Tensor E({kM, kN}, allocator);

    kaleido::core::Tensor acc1({kM, kN}, allocator);
    kaleido::core::Tensor acc2({kM, kN}, allocator);
    kaleido::core::Tensor ref_E({kM, kN}, allocator);

    fill(A, 0, 1e-3);
    fill(B, 0, 1e-3);
    fill(C, 0, 1e-3);
    fill(D, 0, 1e-3);
#ifdef DEBUG
    fill(A, "seq");
    fill(B, "seq");
#endif

    fill(acc1, 0.);
    fill(acc2, 0.);
    fill(E, 0.);
    fill(ref_E, 0.);

    using GemmAddSigmod =
        cuda_kernel::CuteGemmAddSigmod<Element, InstructionShape, ValueMnk,
                                       WarpArrangement, CtaTileShape,
                                       WholeShape>;
    GemmAddSigmod gemm_add_sigmod;
    gemm_add_sigmod(A.data<Element>(), B.data<Element>(), C.data<Element>(),
                    D.data<Element>(), E.mutable_data<Element>());

    cuda_kernel::Cublas2GemmAddSigmod<Element> gemm_add_sigmod_reference;
    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));

    gemm_add_sigmod_reference(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, kN, kM, kK, B.data<Element>(),
        B.dim_size(0), A.data<Element>(), A.dim_size(1), D.data<Element>(),
        ref_E.dim_size(1), C.data<Element>(), ref_E.mutable_data<Element>());

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(E, ref_E, 5e-3);

    std::cout << "Passed unittest for [" << kM << ", " << kN << ", " << kK
              << "]" << std::endl;
}

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_gemm_add_tanh() {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    cudaDeviceProp m_dev_prop;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&m_dev_prop, device_idx);

    // GEMM shape
    int kM = dim_size<0, WholeShape>;
    int kN = dim_size<1, WholeShape>;
    int kK = dim_size<2, WholeShape>;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor A({kM, kK}, allocator);
    kaleido::core::Tensor B({kK, kN}, allocator);
    kaleido::core::Tensor C({kM, kK}, allocator);
    kaleido::core::Tensor D({kK, kN}, allocator);
    kaleido::core::Tensor E({kM, kN}, allocator);

    kaleido::core::Tensor acc1({kM, kN}, allocator);
    kaleido::core::Tensor acc2({kM, kN}, allocator);
    kaleido::core::Tensor ref_E({kM, kN}, allocator);

    fill(A, 0, 1e-3);
    fill(B, 0, 1e-3);
    fill(C, 0, 1e-3);
    fill(D, 0, 1e-3);
#ifdef DEBUG
    fill(A, "seq");
    fill(B, "seq");
#endif

    fill(acc1, 0.);
    fill(acc2, 0.);
    fill(E, 0.);
    fill(ref_E, 0.);

    using GemmAddTanh =
        cuda_kernel::CuteGemmAddTanh<Element, InstructionShape, ValueMnk,
                                     WarpArrangement, CtaTileShape, WholeShape>;
    GemmAddTanh gemm_add_tanh;
    gemm_add_tanh(A.data<Element>(), B.data<Element>(), C.data<Element>(),
                  D.data<Element>(), E.mutable_data<Element>());

    cuda_kernel::Cublas2GemmAddTanh<Element> gemm_add_tanh_reference;
    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));

    gemm_add_tanh_reference(handle, CUBLAS_OP_T, CUBLAS_OP_N, kN, kM, kK,
                            B.data<Element>(), B.dim_size(0),  // k
                            A.data<Element>(),
                            A.dim_size(1),                         // k
                            D.data<Element>(), ref_E.dim_size(1),  // n
                            C.data<Element>(), ref_E.mutable_data<Element>());

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(E, ref_E, 5e-3);

    std::cout << "Passed unittest for [" << kM << ", " << kN << ", " << kK
              << "]" << std::endl;
}

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_lstm_gate() {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    cudaDeviceProp m_dev_prop;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&m_dev_prop, device_idx);

    // GEMM shape
    int kM = dim_size<0, WholeShape>;
    int kN = dim_size<1, WholeShape>;
    int kK = dim_size<2, WholeShape>;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor W({kM, kK}, allocator);
    kaleido::core::Tensor x_t({kK, kN}, allocator);
    kaleido::core::Tensor U({kM, kK}, allocator);
    kaleido::core::Tensor h_t_1({kK, kN}, allocator);
    kaleido::core::Tensor O({kM, kN}, allocator);
    kaleido::core::Tensor ref_O({kM, kN}, allocator);

    fill(W, 0, 1e-3);
    fill(x_t, 0, 1e-3);
    fill(U, 0, 1e-3);
    fill(h_t_1, 0, 1e-3);
#ifdef DEBUG
    // fill(A, "seq");
    // fill(B, "seq");
#endif

    fill(O, 0.);
    fill(ref_O, 0.);

    using LSTMGate =
        cuda_kernel::CuteLSTMGate<Element, InstructionShape, ValueMnk,
                                  WarpArrangement, CtaTileShape, WholeShape>;
    LSTMGate lstm_gate;
    lstm_gate(W.data<Element>(), x_t.data<Element>(), U.data<Element>(),
              h_t_1.data<Element>(), O.mutable_data<Element>());

    cuda_kernel::CublasLSTMGate<Element> lstm_gate_reference;
    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));

    lstm_gate_reference(handle, kM / 4, kN, kK, W.data<Element>(),
                        x_t.data<Element>(), U.data<Element>(),
                        h_t_1.data<Element>(), ref_O.mutable_data<Element>());

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(O, ref_O, 5e-3);

    std::cout << "Passed unittest for [" << kM << ", " << kN << ", " << kK
              << "]" << std::endl;
}

template <typename Element>
void run_lstm_elementwise(int M, int N) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    cudaDeviceProp m_dev_prop;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&m_dev_prop, device_idx);

    ops::FillOp<GPUContext, CUDAPlace, float> fill;
    ops::PrintOp<GPUContext, CUDAPlace, float> printer;

    kaleido::core::Tensor I_t({M, N}, allocator);
    kaleido::core::Tensor F_t({M, N}, allocator);
    kaleido::core::Tensor O_t({M, N}, allocator);
    kaleido::core::Tensor C_t_bar({M, N}, allocator);
    kaleido::core::Tensor C_t_1({M, N}, allocator);
    kaleido::core::Tensor C_t({M, N}, allocator);
    kaleido::core::Tensor H_t({M, N}, allocator);

    kaleido::core::Tensor ref_C_t({M, N}, allocator);
    kaleido::core::Tensor ref_H_t({M, N}, allocator);

    fill(I_t, 0, 1e-3);
    fill(F_t, 0, 1e-3);
    fill(O_t, 0, 1e-3);
    fill(C_t_bar, 0, 1e-3);
    fill(C_t_1, 0, 1e-3);

    fill(C_t, 0.);
    fill(H_t, 0.);
    fill(ref_C_t, 0.);
    fill(ref_H_t, 0.);

    auto lstm_element_wise = &cuda_kernel::KeLSTMElementWise<Element>;

    int THREAD_SIZE = 32 * 32;
    int SIZE = M * N;
    int BLOCK_SIZE = (SIZE + THREAD_SIZE - 1) / THREAD_SIZE;

    dim3 gridDim(BLOCK_SIZE, 1, 1);
    dim3 blockDim(THREAD_SIZE, 1, 1);

    const int kMaxSmemPerBlock = 48 * 1024;
    int smem_size = 6 * THREAD_SIZE * sizeof(Element);

    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(lstm_element_wise,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }

    lstm_element_wise<<<gridDim, blockDim, smem_size>>>(
        I_t.data<Element>(), F_t.data<Element>(), O_t.data<Element>(),
        C_t_bar.data<Element>(), C_t_1.data<Element>(),
        C_t.mutable_data<Element>(), H_t.mutable_data<Element>(), THREAD_SIZE,
        SIZE);

    cuda_kernel::CpuLSTMElementWise<Element> lstm_element_wise_reference;
    lstm_element_wise_reference(
        I_t.data<Element>(), F_t.data<Element>(), O_t.data<Element>(),
        C_t_bar.data<Element>(), C_t_1.data<Element>(),
        ref_C_t.mutable_data<Element>(), ref_H_t.mutable_data<Element>(), M, N);

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(C_t, ref_C_t, 5e-3);
    check(H_t, ref_H_t, 5e-3);

    std::cout << "Passed unittest for [" << M << ", " << N << "]" << std::endl;
}

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_cute_lstm_cell() {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    cudaDeviceProp m_dev_prop;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&m_dev_prop, device_idx);

    // GEMM shape
    int kM = dim_size<0, WholeShape>;
    int kN = dim_size<1, WholeShape>;
    int kK = dim_size<2, WholeShape>;

    int M = kM / 4;
    int N = kN;
    int K = kK;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor W({kM, kK}, allocator);
    kaleido::core::Tensor x_t({kK, kN}, allocator);
    kaleido::core::Tensor U({kM, kK}, allocator);
    kaleido::core::Tensor h_t_1({kK, kN}, allocator);
    kaleido::core::Tensor O({kM, kN}, allocator);
    kaleido::core::Tensor ref_O({kM, kN}, allocator);

    kaleido::core::Tensor C_t_1({M, N}, allocator);
    kaleido::core::Tensor C_t({M, N}, allocator);
    kaleido::core::Tensor H_t({M, N}, allocator);
    kaleido::core::Tensor ref_C_t({M, N}, allocator);
    kaleido::core::Tensor ref_H_t({M, N}, allocator);

    fill(W, 0, 1e-3);
    fill(x_t, 0, 1e-3);
    fill(U, 0, 1e-3);
    fill(h_t_1, 0, 1e-3);

    fill(C_t_1, 0, 1e-3);
#ifdef DEBUG
    // fill(A, "seq");
    // fill(B, "seq");
#endif

    fill(O, 0.);
    fill(ref_O, 0.);

    fill(C_t, 0.);
    fill(H_t, 0.);
    fill(ref_C_t, 0.);
    fill(ref_H_t, 0.);

    using LSTMGate =
        cuda_kernel::CuteLSTMGate<Element, InstructionShape, ValueMnk,
                                  WarpArrangement, CtaTileShape, WholeShape>;
    LSTMGate lstm_gate;
    lstm_gate(W.data<Element>(), x_t.data<Element>(), U.data<Element>(),
              h_t_1.data<Element>(), O.mutable_data<Element>());

    cuda_kernel::CublasLSTMGate<Element> lstm_gate_reference;
    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));

    lstm_gate_reference(handle, M, N, K, W.data<Element>(), x_t.data<Element>(),
                        U.data<Element>(), h_t_1.data<Element>(),
                        ref_O.mutable_data<Element>());

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(O, ref_O, 5e-3);

    const Element* I_t = O.data<Element>();
    const Element* F_t = O.data<Element>() + M * N;
    const Element* O_t = O.data<Element>() + 2 * M * N;
    const Element* C_t_bar = O.data<Element>() + 3 * M * N;

    auto lstm_element_wise = &cuda_kernel::KeLSTMElementWise<Element>;
    int THREAD_SIZE = 32 * 32;
    int SIZE = M * N;
    int BLOCK_SIZE = (SIZE + THREAD_SIZE - 1) / THREAD_SIZE;

    dim3 gridDim(BLOCK_SIZE, 1, 1);
    dim3 blockDim(THREAD_SIZE, 1, 1);

    const int kMaxSmemPerBlock = 48 * 1024;
    static int smem_size = 6 * THREAD_SIZE * sizeof(Element);

    if (smem_size > kMaxSmemPerBlock) {
        cudaFuncSetAttribute(lstm_element_wise,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_size);
    }

    lstm_element_wise<<<gridDim, blockDim, smem_size>>>(
        I_t, F_t, O_t, C_t_bar, C_t_1.data<Element>(),
        C_t.mutable_data<Element>(), H_t.mutable_data<Element>(), THREAD_SIZE,
        SIZE);

    cudaDeviceSynchronize();

    cuda_kernel::CpuLSTMElementWise<Element> lstm_element_wise_reference;
    lstm_element_wise_reference(I_t, F_t, O_t, C_t_bar, C_t_1.data<Element>(),
                                ref_C_t.mutable_data<Element>(),
                                ref_H_t.mutable_data<Element>(), M, N);

    check(C_t, ref_C_t, 5e-3);
    check(H_t, ref_H_t, 5e-3);

    std::cout << "Passed unittest for [" << kM << ", " << kN << ", " << kK
              << "]" << std::endl;
}

template <typename Element, typename WarpArrangement, typename CtaTileShape,
          typename WholeShape, typename InstructionShape = TileShape<16, 8, 16>,
          typename ValueMnk = TileShape<1, 2, 1>>
void run_whole_cute_lstm_cell() {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);
    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    cudaDeviceProp m_dev_prop;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&m_dev_prop, device_idx);

    // GEMM shape
    int kM = dim_size<0, WholeShape>;
    int kN = dim_size<1, WholeShape>;
    int kK = dim_size<2, WholeShape>;

    int M = kM / 4;
    int N = kN;
    int K = kK;

    // ThreadBlock tile shape
    int kTM = dim_size<0, CtaTileShape>;
    int kTN = dim_size<1, CtaTileShape>;
    int kTK = dim_size<2, CtaTileShape>;

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;
    ops::PrintOp<GPUContext, CUDAPlace, Element> printer;

    kaleido::core::Tensor W({kM, kK}, allocator);
    kaleido::core::Tensor x_t({kK, kN}, allocator);
    kaleido::core::Tensor U({kM, kK}, allocator);
    kaleido::core::Tensor h_t_1({kK, kN}, allocator);
    kaleido::core::Tensor O({kM, kN}, allocator);
    kaleido::core::Tensor ref_O({kM, kN}, allocator);

    kaleido::core::Tensor C_t_1({M, N}, allocator);
    kaleido::core::Tensor C_t({M, N}, allocator);
    kaleido::core::Tensor H_t({M, N}, allocator);
    kaleido::core::Tensor ref_C_t({M, N}, allocator);
    kaleido::core::Tensor ref_H_t({M, N}, allocator);

    fill(W, 0, 1e-3);
    fill(x_t, 0, 1e-3);
    fill(U, 0, 1e-3);
    fill(h_t_1, 0, 1e-3);

    fill(C_t_1, 0, 1e-3);
#ifdef DEBUG
    // fill(A, "seq");
    // fill(B, "seq");
#endif
    fill(O, 0.);
    fill(ref_O, 0.);

    fill(C_t, 0.);
    fill(H_t, 0.);
    fill(ref_C_t, 0.);
    fill(ref_H_t, 0.);

    using LSTMCell =
        cuda_kernel::CuteLSTMCell<Element, InstructionShape, ValueMnk,
                                  WarpArrangement, CtaTileShape, WholeShape>;
    LSTMCell lstm_cell;
    lstm_cell(W.data<Element>(), x_t.data<Element>(), U.data<Element>(),
              C_t_1.data<Element>(), h_t_1.data<Element>(),
              C_t.mutable_data<Element>(), H_t.mutable_data<Element>(),
              O.mutable_data<Element>());

    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));

    cuda_kernel::ReferenceLSTMCell<Element> lstm_cell_reference;
    lstm_cell_reference(handle, W.data<Element>(), x_t.data<Element>(),
                        U.data<Element>(), C_t_1.data<Element>(),
                        h_t_1.data<Element>(), ref_C_t.mutable_data<Element>(),
                        ref_H_t.mutable_data<Element>(),
                        ref_O.mutable_data<Element>(), M, N, K);

    CublasCheck(cublasDestroy(handle));

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;
    check(O, ref_O, 5e-3);

    check(C_t, ref_C_t, 5e-3);
    check(H_t, ref_H_t, 5e-3);

    std::cout << "Passed unittest for [" << kM << ", " << kN << ", " << kK
              << "]" << std::endl;
}

TEST(Test2GemmAdd, test_cute_2gemm_add) {
    // The minimal valid shape
    run_cute_2gemm_add<cutlass::half_t, TileShape<1, 1, 1>, /*Warp
    Arrangement*/
                       TileShape<16, 32, 32>,               /*CTA Shape*/
                       TileShape<4 * 16, 32, 32>            /*Whole Shape*/
                       >();

    run_cute_2gemm_add<cutlass::half_t, TileShape<1, 1, 1>, /*Warp
    Arrangement*/
                       TileShape<16, 32, 32>,               /*CTA Shape*/
                       TileShape<16, 32, 64>                /*Whole Shape*/
                       >();

    run_cute_2gemm_add<cutlass::half_t, TileShape<2, 2, 1>, /*Warp
    Arrangement*/
                       TileShape<64, 128, 64>,              /*CTA Shape*/
                       TileShape<128, 256, 512>             /*Whole Shape*/
                       >();
}

TEST(TestGemmAddSigmod, test_cute_gemm_add_sigmod) {
    // The minimal valid shape
    run_cute_gemm_add_sigmod<cutlass::half_t,
                             TileShape<1, 1, 1>,    /*Warp Arrangement*/
                             TileShape<16, 32, 32>, /*CTA Shape*/
                             TileShape<16, 32, 32>  /*Whole Shape*/
                             >();
    run_cute_gemm_add_sigmod<cutlass::half_t,
                             TileShape<1, 1, 1>,    /*Warp Arrangement*/
                             TileShape<16, 32, 32>, /*CTA Shape*/
                             TileShape<16, 32, 64>  /*Whole Shape*/
                             >();

    run_cute_gemm_add_sigmod<cutlass::half_t,
                             TileShape<2, 2, 1>,      /*Warp Arrangement*/
                             TileShape<64, 128, 64>,  /*CTA Shape*/
                             TileShape<128, 256, 512> /*Whole Shape*/
                             >();
}

TEST(TestGemmAddTanh, test_cute_gemm_add_tanh) {
    // The minimal valid shape
    run_cute_gemm_add_tanh<cutlass::half_t,
                           TileShape<1, 1, 1>,    /*Warp Arrangement*/
                           TileShape<16, 32, 32>, /*CTA Shape*/
                           TileShape<16, 32, 32>  /*Whole Shape*/
                           >();
    run_cute_gemm_add_tanh<cutlass::half_t,
                           TileShape<1, 1, 1>,    /*Warp Arrangement*/
                           TileShape<16, 32, 32>, /*CTA Shape*/
                           TileShape<16, 32, 64>  /*Whole Shape*/
                           >();

    run_cute_gemm_add_tanh<cutlass::half_t,
                           TileShape<2, 2, 1>,      /*Warp Arrangement*/
                           TileShape<64, 128, 64>,  /*CTA Shape*/
                           TileShape<128, 256, 512> /*Whole Shape*/
                           >();
}

TEST(TestLSTMGate, test_cute_lstm_gate) {
    // The minimal valid shape
    run_cute_lstm_gate<cutlass::half_t, TileShape<1, 1, 1>, /*Warp
    Arrangement*/
                       TileShape<16, 32, 32>,               /*CTA Shape*/
                       TileShape<4 * 16, 32, 32>            /*Whole Shape*/
                       >();
    run_cute_lstm_gate<cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
                       TileShape<16, 32, 32>,               /*CTA Shape*/
                       TileShape<4 * 16, 32, 64>            /*Whole Shape*/
                       >();

    run_cute_lstm_gate<cutlass::half_t, TileShape<2, 2, 1>, /*Warp Arrangement*/
                       TileShape<64, 128, 64>,              /*CTA Shape*/
                       TileShape<4 * 128, 256, 512>         /*Whole Shape*/
                       >();
}

TEST(TestLSTMElementWise, test_lstm_element_wise) {
    run_lstm_elementwise<float>(32, 32);
    run_lstm_elementwise<float>(64, 64);
    run_lstm_elementwise<float>(128, 128);
    run_lstm_elementwise<float>(256, 256);
    run_lstm_elementwise<float>(512, 512);
}

TEST(TestLSTMCell, test_cute_lstm_cell) {
    // The minimal valid shape
    run_cute_lstm_cell<cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
                       TileShape<16, 32, 32>,               /*CTA Shape*/
                       TileShape<4 * 16, 32, 32>            /*Whole Shape*/
                       >();

    run_cute_lstm_cell<cutlass::half_t, TileShape<1, 1, 1>, /*Warp Arrangement*/
                       TileShape<16, 32, 32>,               /*CTA Shape*/
                       TileShape<4 * 16, 32, 64>            /*Whole Shape*/
                       >();

    run_cute_lstm_cell<cutlass::half_t, TileShape<2, 2, 1>, /*Warp Arrangement*/
                       TileShape<64, 128, 64>,              /*CTA Shape*/
                       TileShape<4 * 128, 256, 512>         /*Whole Shape*/
                       >();
}

TEST(TestLSTMCell, test_whole_cute_lstm_cell) {
    // The minimal valid shape
    // Whole Shape: [4 * hidden_size, batch_size, hidden_size]
    run_whole_cute_lstm_cell<cutlass::half_t,
                             TileShape<1, 1, 1>,          /*Warp Arrangement*/
                             TileShape<16, 32, 32>,       /*CTA Shape*/
                             TileShape<4 * 128, 256, 128> /*Whole Shape*/
                             >();

    run_whole_cute_lstm_cell<cutlass::half_t,
                             TileShape<1, 1, 1>,          /*Warp Arrangement*/
                             TileShape<16, 32, 32>,       /*CTA Shape*/
                             TileShape<4 * 256, 256, 256> /*Whole Shape*/
                             >();

    run_whole_cute_lstm_cell<cutlass::half_t,
                             TileShape<1, 1, 1>,          /*Warp Arrangement*/
                             TileShape<16, 32, 32>,       /*CTA Shape*/
                             TileShape<4 * 512, 256, 512> /*Whole Shape*/
                             >();

    run_whole_cute_lstm_cell<cutlass::half_t,
                             TileShape<1, 1, 1>,            /*Warp Arrangement*/
                             TileShape<16, 32, 32>,         /*CTA Shape*/
                             TileShape<4 * 1024, 256, 1024> /*Whole Shape*/
                             >();

    run_whole_cute_lstm_cell<cutlass::half_t,
                             TileShape<1, 1, 1>,            /*Warp Arrangement*/
                             TileShape<16, 32, 32>,         /*CTA Shape*/
                             TileShape<4 * 2048, 256, 2048> /*Whole Shape*/
                             >();
}

}  // namespace core
}  // namespace kaleido
