#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/gemm.h"
#include "kaleido/core/operators/matmul_op.h"
#include "kaleido/core/types.h"

#include <cutlass/numeric_types.h>
#include <glog/logging.h>

namespace kaleido {
namespace core {
namespace ops {

template <typename T>
class MatMulOp<GPUContext, CUDAPlace, T> {
   public:
    void operator()(const GPUContext& context, const Tensor& A, bool trans_a,
                    const Tensor& B, bool trans_b, Tensor& C, T alf = 1.,
                    T bet = 0.) {
        // TODO(ying): all checks can be moved to compile-time in future.
        CHECK((A.layout() == B.layout() && A.layout() == C.layout()))
            << "A, B, C should have the same layout. "
            << "Otherwise, not implemented yet.";

        CHECK_LE(A.ndim(), 2);
        CHECK_LE(B.ndim(), 2);
        CHECK_LE(C.ndim(), 2);

        const T* alpha = &alf;
        const T* beta = &bet;

        cublasHandle_t handle = context.cublas_handle();

        int m = trans_a ? A.dim_size(1) : A.dim_size(0);  // row of the output
        int n =
            trans_b ? B.dim_size(0) : B.dim_size(1);  // column of the output
        int k = trans_a ? A.dim_size(0) : A.dim_size(1);  // the contraction dim

        cublasOperation_t transa = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t transb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

        cuda_kernel::CublasGemm<T> gemm;
        switch (A.layout()) {
            case Layout::COL_MAJOR: {
                // cuBlas is a Fortran-style(column-major) BLAS library.
                // When A, B, C are lay out in colum major, such a matrix
                // multiplcation can be directly mapped to cuBlas gemm.
                int lda = A.dim_size(0);
                int ldb = B.dim_size(0);
                int ldc = C.dim_size(0);

                gemm(handle, transa, transb, m, n, k, alpha, A.data<T>(), lda,
                     B.data<T>(), ldb, beta, C.mutable_data<T>(), ldc);

                break;
            }
            case Layout::ROW_MAJOR: {
                // cuBlas is a Fortran-style(column-major) BLAS library.
                // When A, B, C are lay out in row major,
                // slyly call cublas as it compute C^T = (AB)^T = (B^T)(A^T).

                int lda = A.dim_size(1);
                int ldb = B.dim_size(1);
                int ldc = C.dim_size(1);

                gemm(handle, transb, transa, n, m, k, alpha, B.data<T>(), ldb,
                     A.data<T>(), lda, beta, C.mutable_data<T>(), ldc);

                break;
            }
            default:
                LOG(FATAL) << "Error layout." << std::endl;
        }
    }
};

template class MatMulOp<GPUContext, CUDAPlace, float>;
template class MatMulOp<GPUContext, CUDAPlace, __half>;
template class MatMulOp<GPUContext, CUDAPlace, cutlass::half_t>;

}  // namespace ops
}  // namespace core
}  // namespace kaleido
