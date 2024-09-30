#pragma once

namespace kaleido {
namespace core {
namespace ops {

inline bool IsPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

inline unsigned int NextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

inline unsigned int Log2Floor(unsigned int x) {
    if (x == 0) return -1U;
    int log = 0;
    unsigned int value = x;
    for (int i = 4; i >= 0; --i) {
        int shift = (1 << i);
        unsigned int n = value >> shift;
        if (n != 0) {
            value = n;
            log += shift;
        }
    }
    assert(value == 1);
    return log;
}

template <typename T, typename X, typename Y>
inline T DivUp(const X x, const Y y) {
    return static_cast<T>((x + y - 1) / y);
}

void GetGpuLaunchConfig1D(const GPUContext& ctx, int64_t numel, int* threads,
                          int* blocks) {
    int num_threads = ctx.GetMaxThreadsPerBlock();
    int sm_count = ctx.GetSMCount();

    if (numel / (sm_count << 1) < num_threads)
        num_threads = NextPow2(numel / (sm_count << 1));
    else if (numel / (sm_count << 2) < num_threads)
        num_threads = NextPow2(numel / (sm_count << 2));

    *threads = std::max(64, num_threads);
    *blocks = DivUp<int, int, int>(numel, *threads);
}

}  // namespace ops
}  // namespace core
}  // namespace kaleido
