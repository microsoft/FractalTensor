#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/operators/gemm_batched_op.h"
#include "kaleido/core/types.h"

#include <cublas_v2.h>
#include <glog/logging.h>

namespace kaleido {
namespace core {
namespace ops {

namespace {

template <typename T>
struct CublasGemmBatched {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const T* alpha, const T* Aarray[], int lda,
                    const T* Barray[], int ldb, const T* beta, T* Carray[],
                    int ldc, int batchCount);
};

template <>
struct CublasGemmBatched<float> {
    void operator()(cublasHandle_t handle, cublasOperation_t transa,
                    cublasOperation_t transb, int m, int n, int k,
                    const float* alpha, const float* Aarray[], int lda,
                    const float* Barray[], int ldb, const float* beta,
                    float* Carray[], int ldc, int batchCount) {
        CublasCheck(cublasSgemmBatched(handle, transa, transb, m, n, k, alpha,
                                       Aarray, lda, Barray, ldb, beta, Carray,
                                       ldc, batchCount));
    }
};

}  // namespace

template <typename T>
class GemmBatchedOp<GPUContext, CUDAPlace, T> {
   public:
    void operator()(const GPUContext& context, const std::vector<Tensor>& A,
                    bool trans_a, const std::vector<Tensor>& B, bool trans_b,
                    std::vector<Tensor>& C, T alf = 1., T bet = 0.) {
        CHECK(A.size() == B.size() && A.size() == C.size())
            << "A, B, C should have a same number of matrices.";

        int batch_count = A.size();

        for (int i = 0; i < A.size(); ++i) {
            CHECK(A[i].layout() == B[i].layout() &&
                  A[i].layout() == C[i].layout())
                << "A, B, C should have the same layout. "
                << "Otherwise, not implemented yet.";
            CHECK_LE(A[i].ndim(), 2);
            CHECK_LE(B[i].ndim(), 2);
            CHECK_LE(C[i].ndim(), 2);
        }

        const T* alpha = &alf;
        const T* beta = &bet;

        const T* dAarrayTmp[batch_count];
        const T* dBarrayTmp[batch_count];
        T* dCarrayTmp[batch_count];

        for (int i = 0; i < batch_count; ++i) {
            dAarrayTmp[i] = A[i].data<T>();
            dBarrayTmp[i] = B[i].data<T>();
            dCarrayTmp[i] = C[i].mutable_data<T>();
        }

        T **devAarray, **devBarray, **devCarray;
        CudaCheck(cudaMalloc(&devAarray, batch_count * sizeof(T*)));
        CudaCheck(cudaMemcpy(devAarray, dAarrayTmp, batch_count * sizeof(T*),
                             cudaMemcpyHostToDevice));
        CudaCheck(cudaMalloc(&devBarray, batch_count * sizeof(T*)));
        CudaCheck(cudaMemcpy(devBarray, dBarrayTmp, batch_count * sizeof(T*),
                             cudaMemcpyHostToDevice));
        CudaCheck(cudaMalloc(&devCarray, batch_count * sizeof(T*)));
        CudaCheck(cudaMemcpy(devCarray, dCarrayTmp, batch_count * sizeof(T*),
                             cudaMemcpyHostToDevice));

        int m = trans_a ? A[0].dim_size(1) : A[0].dim_size(0);
        int n = trans_b ? B[0].dim_size(0) : B[0].dim_size(1);
        int k = trans_a ? A[0].dim_size(0) : A[0].dim_size(1);

        cublasOperation_t transa = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t transb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

        CublasGemmBatched<T> gemm;
        cublasHandle_t handle = context.cublas_handle();
        switch (A[0].layout()) {
            case Layout::COL_MAJOR: {
                int lda = A[0].dim_size(0);
                int ldb = B[0].dim_size(0);
                int ldc = C[0].dim_size(0);

                gemm(handle, transa, transb, m, n, k, alpha,
                     (const T**)devAarray, lda, (const T**)devBarray, ldb, beta,
                     (T**)devCarray, ldc, batch_count);
                break;
            }
            case Layout::ROW_MAJOR: {
                int lda = A[0].dim_size(1);
                int ldb = B[0].dim_size(1);
                int ldc = C[0].dim_size(1);

                gemm(handle, transb, transa, n, m, k, alpha,
                     (const T**)devBarray, ldb, (const T**)devAarray, lda, beta,
                     (T**)devCarray, ldc, batch_count);
                break;
            }
            default:
                LOG(FATAL) << "Error layout." << std::endl;
        }
        cudaFree(devAarray);
        cudaFree(devBarray);
        cudaFree(devCarray);
    }
};

template class GemmBatchedOp<GPUContext, CUDAPlace, float>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
