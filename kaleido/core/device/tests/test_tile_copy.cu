#include "kaleido/core/device/kernels/tiled_copy.h"

#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace kaleido {
namespace core {

using namespace kaleido::core::cuda_kernel;

namespace {

void CheckResult(const __half* data1, const __half* data2, int numel) {
    float abs_error = 1e-5;
    for (int i = 0; i < numel; ++i) {
        EXPECT_NEAR(__half2float(data1[i]), __half2float(data2[i]), abs_error)
            << "i: " << i << ", data1: " << __half2float(data1[i])
            << ", data2: " << __half2float(data2[i]) << std::endl;
    }
}

template <typename Element, typename G2S, typename S2G, const int N>
__global__ void TestCopy(const Element* data1, G2S& loader, Element* data2,
                         S2G& storer) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* smem_buf = reinterpret_cast<Element*>(shared_buf);

    loader.copy(data1, smem_buf, threadIdx.x);
    commit_copy_group();
    wait_group<0>();
    __syncthreads();

    storer.copy(smem_buf, data2, threadIdx.x);
    __syncthreads();
}

}  // namespace

TEST(TestTileCopy, test) {
    using Element = __half;

    static const int kRow = 32;
    static const int kCol = 64;

    const int numel = kRow * kCol;

    thrust::host_vector<Element> h_A(numel);
    srand(42);
    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = __float2half(i);
    }
    thrust::device_vector<Element> d_A = h_A;
    thrust::device_vector<Element> d_B(numel);
    thrust::fill(d_B.begin(), d_B.end(), static_cast<Element>(0.));

    using ThreadsShape = TileShape<16, 8>;
    using G2S = G2SCopy2D<Element, ThreadsShape, RowMajor<kRow, kCol>,
                          RowMajor<kRow, kCol>>;
    G2S loader;
    using S2G = S2GCopy2D<Element, ThreadsShape, RowMajor<kRow, kCol>,
                          RowMajor<kRow, kCol>>;
    S2G storer;

    int shm_size = numel * sizeof(Element);
    const int kThreads = get_numel<ThreadsShape>;
    TestCopy<Element, G2S, S2G, numel><<<1, kThreads, shm_size, 0>>>(
        thrust::raw_pointer_cast(d_A.data()), loader,
        thrust::raw_pointer_cast(d_B.data()), storer);
    cudaDeviceSynchronize();

    thrust::host_vector<Element> h_B(numel);
    h_B = d_B;

    CheckResult(thrust::raw_pointer_cast(h_A.data()),
                thrust::raw_pointer_cast(h_B.data()), numel);
}

}  // namespace core
}  // namespace kaleido
