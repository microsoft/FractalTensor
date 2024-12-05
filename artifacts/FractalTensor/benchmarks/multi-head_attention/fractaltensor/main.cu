#include "kaleido/core/device/cuda_timer.h"
#include "kernel.h"
#include "utils/kernel_traits.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace {
float rand_float(float a = 1e-1, float b = 5e-2) {
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}
}  // namespace

template <const int length, const int hidden_qk, const int hidden_v,
          const int batch, const int head, const int kTileSizeRow = 64,
          int kTileSizeCol = 64, const int kBlockKSmem = 64,
          const int kThreads = 128, const int stages_qk = 1,
          const int stages_v = 1>
void run_test() {
    using InType = cutlass::half_t;
    using AccType = float;

    thrust::host_vector<InType> h_q(batch * head * length * hidden_qk);
    for (int i = 0; i < h_q.size(); ++i)
        h_q[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_k(batch * head * length * hidden_qk);
    for (int i = 0; i < h_k.size(); ++i)
        h_k[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_v(batch * head * length * hidden_v);
    for (int i = 0; i < h_v.size(); ++i)
        h_v[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_o(batch * head * length * hidden_v);
    for (int i = 0; i < h_o.size(); ++i)
        h_o[i] = static_cast<InType>(rand_float());

    thrust::device_vector<InType> d_q = h_q;
    thrust::device_vector<InType> d_k = h_k;
    thrust::device_vector<InType> d_v = h_v;
    thrust::device_vector<InType> d_o = h_o;

    const int kDimK = hidden_qk;
    const int kDimV = hidden_v;

    const bool load_q_once = (kBlockKSmem == kDimK);
    constexpr const int SmemKAtom = 64;
    constexpr const int kSwizzle = SmemKAtom == 32 ? 2 : 3;

    constexpr int shared_in =
        stages_qk * kTileSizeRow * kBlockKSmem * sizeof(InType) +
        stages_qk * kTileSizeCol * kBlockKSmem * sizeof(InType) +
        stages_v * kBlockKSmem * kDimV * sizeof(InType);

    constexpr int shared_out = kTileSizeRow * kDimV * sizeof(InType);
    constexpr int shared_mem = shared_in > shared_out ? shared_in : shared_out;

    using Traits = KeTraits<InType, kThreads, kDimK, kDimV, kTileSizeRow,
                            kTileSizeCol, kBlockKSmem, stages_qk, load_q_once,
                            kBlockKSmem, stages_v, SmemKAtom, kSwizzle>;

    auto kernel =
        &flashattn<InType, AccType, Traits, kDimK, kDimV, kTileSizeRow,
                   kTileSizeCol, kThreads, kBlockKSmem, stages_qk, load_q_once,
                   kBlockKSmem, stages_v, SmemKAtom, kSwizzle>;

    if (shared_mem > 48 * 1024)
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem);

    int warm_up = 10;
    for (int i = 0; i < warm_up; ++i) {
        kernel<<<dim3(batch * head * length / kTileSizeRow, 1, 1),
                 dim3(kThreads, 1, 1), shared_mem, 0>>>(
            thrust::raw_pointer_cast(d_q.data()),
            thrust::raw_pointer_cast(d_k.data()),
            thrust::raw_pointer_cast(d_v.data()),
            thrust::raw_pointer_cast(d_o.data()), length, length);
    }

    int iter = 20;
    kaleido::core::CudaTimer timer;
    timer.Start();
    for (int i = 0; i < iter; ++i) {
        kernel<<<dim3(batch * head * length / kTileSizeRow, 1, 1),
                 dim3(kThreads, 1, 1), shared_mem, 0>>>(
            thrust::raw_pointer_cast(d_q.data()),
            thrust::raw_pointer_cast(d_k.data()),
            thrust::raw_pointer_cast(d_v.data()),
            thrust::raw_pointer_cast(d_o.data()), length, length);
    }
    cudaDeviceSynchronize();
    float time = timer.Stop() / iter;

    std::cout << length << "\t" << hidden_qk << "\t" << time << std::endl;
}

int main() {
    // length, hidden_q, hidden_v, batch, head,
    run_test<128, 128, 128, 32, 8>();
    run_test<256, 128, 128, 32, 8>();
    run_test<512, 128, 128, 32, 8>();
    run_test<768, 128, 128, 32, 8>();
    run_test<1024, 128, 128, 32, 8>();
    run_test<1536, 128, 128, 32, 8>();
    run_test<2048, 128, 128, 32, 8>();
    run_test<4096, 128, 128, 32, 8>();

    run_test<128, 256, 256, 32, 8>();
    run_test<256, 256, 256, 32, 8>();
    run_test<512, 256, 256, 32, 8>();
    run_test<768, 256, 256, 32, 8>();
    run_test<1024, 256, 256, 32, 8>();
    run_test<1536, 256, 256, 32, 8>();
    run_test<2048, 256, 256, 32, 8>();
    run_test<4096, 256, 256, 32, 8>();

    return 0;
}
