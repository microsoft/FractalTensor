#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_timer.h"
#include "kaleido/core/device/cuda_utils.h"
#include "kaleido/core/device/gpu_context.h"
#include "kaleido/core/device/kernels/gemm.h"
#include "kaleido/core/operators/expect_eq_op.h"
#include "kaleido/core/operators/fill_op.h"
#include "kaleido/core/operators/tests/b2b_gemm_test_utils.h"
#include "kaleido/core/place.h"
#include "kaleido/core/tensor.h"
#include "kaleido/core/tile_shape.h"

#include <glog/logging.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace kaleido::core;

template <typename Element, typename WholeShape, typename CtaTileShape,
          typename WarpShape>
void run_test(std::ofstream& fout) {
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

    ops::FillOp<GPUContext, CUDAPlace, Element> fill;

    const int kM = dim_size<0, WholeShape>;
    const int kN = dim_size<1, WholeShape>;
    const int kK = dim_size<2, WholeShape>;
    const int kP = dim_size<3, WholeShape>;

    const int kTM = dim_size<0, CtaTileShape>;
    const int kTN = dim_size<1, CtaTileShape>;
    const int kTK = dim_size<2, CtaTileShape>;
    const int kTP = dim_size<3, CtaTileShape>;

    int shm_input =
        (kTM * kTK /*A tile*/ + kTK * kTN /*B tile*/ + kTN * kTP /*B tile*/);
    int shm_output = kTM * kTP /*output tile*/;

    // output tile reuse the shared memory buffer for the input tiles
    int shm_size = shm_input < shm_output ? shm_output * sizeof(Element)
                                          : shm_input * sizeof(Element);
    LOG(INFO) << "shared memory size:" << shm_size / 1024 << "KB";

    int num_blocks = CeilDiv<kM, kTM> * CeilDiv<kP, kTP>;
    int num_threads = dim_size<0, WarpShape> * 32;

    fout << "[" << kM << ", " << kK << "][" << kK << ", " << kN << "][" << kN
         << ", " << kP << "]\t";
    fout << "[" << kTM << ", " << kTK << "][" << kTK << ", " << kTN << "]["
         << kTN << ", " << kTP << "]\t" << shm_size / 1024 << "\t" << num_blocks
         << "\t" << num_threads << "\t";

    kaleido::core::Tensor A({kM, kK}, allocator);
    kaleido::core::Tensor B({kK, kN}, allocator);
    kaleido::core::Tensor C({kN, kP}, allocator);
    kaleido::core::Tensor D({kM, kP}, allocator);

    kaleido::core::Tensor ref_P({kM, kN}, allocator);
    kaleido::core::Tensor ref_D({kM, kP}, allocator);

    fill(A, 0., 1e-3);
    fill(B, 0., 1e-3);
    fill(C, 0., 1e-3);

    fill(D, 0.);
    fill(ref_P, 0.);
    fill(ref_D, 0.);

    cublasHandle_t handle;
    CublasCheck(cublasCreate(&handle));

    using Gemm =
        cuda_kernel::B2BGemm<Element, WholeShape, CtaTileShape, WarpShape>;
    Gemm gemm;

    ops::ExpectEqOp<GPUContext, CUDAPlace, Element> check;

    bool passed_unittest = false;

    const int warm_up = 50;
    for (int i = 0; i < warm_up; ++i) {  // warm up
        gemm(A.data<Element>(), B.data<Element>(), C.data<Element>(),
             D.mutable_data<Element>());

        cublas_two_hgemms<Element>(handle, A, B, C,  // inputs
                                   ref_P,            // ref_P = A @ B
                                   ref_D /*ref_D = ref_P @ C*/);

        if (!passed_unittest) {  // check correctness
            check(D, ref_D, 3e-3);
            passed_unittest = true;
        }
    }

    const int iter = 200;

    CudaTimer timer;
    timer.Start();
    for (int i = 0; i < iter; ++i) {
        cublas_two_hgemms<Element>(handle, A, B, C,  // inputs
                                   ref_P,            // ref_P = A @ B
                                   ref_D /*ref_D = ref_P @ C*/);
    }
    float time1 = timer.Stop() / iter;
    CublasCheck(cublasDestroy(handle));

    timer.Start();
    for (int i = 0; i < iter; ++i) {
        gemm(A.data<Element>(), B.data<Element>(), C.data<Element>(),
             D.mutable_data<Element>());
    }
    float time2 = timer.Stop() / iter;

    fout << time1 << "\t" << time2 << "\t" << time2 / time1 << "\n";
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const char* filename = argv[1];

    google::InitGoogleLogging("back-to-back gemms");

    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    std::stringstream file_name;
    file_name << filename;
    fout.open(file_name.str(), std::ios::out);
    fout << "GEMM Shape\tCTA Tile Shape\tShared "
            "Memory(KB)\tblocks\tthreads\tcuBLAS(ms)\tFused two "
            "GEMMs(ms)\tRatio "
            "to cuBLAS\n";

    run_test<cutlass::half_t,
             TileShape<8192 /*kM*/, 256 /*kN*/, 64 /*kK*/, 64 /*kP*/>,
             TileShape<128 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
             TileShape<4, 1> /*b2b gemm requires 1 warp per column*/
             >(fout);

    run_test<cutlass::half_t,
             TileShape<8192 /*kM*/, 512 /*kN*/, 64 /*kK*/, 64 /*kP*/>,
             TileShape<128 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
             TileShape<4, 1> /*b2b gemm requires 1 warp per column*/
             >(fout);

    run_test<cutlass::half_t,
             TileShape<16384 /*kM*/, 256 /*kN*/, 64 /*kK*/, 64 /*kP*/>,
             TileShape<128 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
             TileShape<8, 1> /*b2b gemm requires 1 warp per column*/
             >(fout);

    run_test<cutlass::half_t,
             TileShape<16384 /*kM*/, 512 /*kN*/, 64 /*kK*/, 64 /*kP*/>,
             TileShape<128 /*kTM*/, 64 /*kTN*/, 64 /*kTK*/, 64 /*kTP*/>,
             TileShape<8, 1> /*b2b gemm requires 1 warp per column*/
             >(fout);

    return 0;
}
