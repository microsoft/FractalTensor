#pragma once

/*
  A and D are laid out in row-major fashion
  B and C are laid out in column-major fashion

  A[m, k] @ B[k, n]
  D[m, p] = P[m, n] @ C[n, p]
*/
template <typename Element>
void cublas_two_hgemms(cublasHandle_t& handle, const kaleido::core::Tensor& A,
                       const kaleido::core::Tensor& B,
                       const kaleido::core::Tensor& C, kaleido::core::Tensor& P,
                       kaleido::core::Tensor& D) {
    int kM = A.dim_size(0);
    int kN = B.dim_size(1);
    int kK = A.dim_size(1);
    int kP = C.dim_size(1);

    // cuBLAS gemm as the groundtruth
    kaleido::core::cuda_kernel::CublasGemm<Element> hgemm;

    Element alf = static_cast<Element>(1.);
    Element bet = static_cast<Element>(0.);

    // P = A @ B
    // P^T = B^T @ A^T
    hgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kN, kM, kK, &alf,
          B.data<Element>(), B.dim_size(0), A.data<Element>(), A.dim_size(1),
          &bet, P.mutable_data<Element>(), P.dim_size(1));

    // D = P @ C, D and P are laid out in row-major fashion, while C is in
    // column major fashion. Operands of cuBLAS is by default in column fashion.
    // D^T = C^T @ P^T; [p, m] = [p, n] @ [n, m]
    hgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N /* transb*/, kP, kM, kN, &alf,
          C.data<Element>(), C.dim_size(0), P.data<Element>(), P.dim_size(1),
          &bet, D.mutable_data<Element>(), D.dim_size(1));
}
