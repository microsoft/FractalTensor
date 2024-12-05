#pragma once

#include <cutlass/numeric_conversion.h>

namespace kaleido {
namespace core {
namespace cuda_kernel {

using namespace cute;

template <typename To_type, typename Engine, typename Layout>
CUTE_DEVICE auto convert_type(cute::Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag =
        convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(
            tensor.data()));

    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename Tensor>
struct IndexedTensor_ {
    DEVICE IndexedTensor_(Tensor& tensor) : tensor_(tensor) {}

    DEVICE const auto& operator[](int idx) { return tensor_(_, _, idx); }

   private:
    Tensor& tensor_;
};

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M,
// MMA_N / 2) if using m16n8k16, or to (4, MMA_M, MMA_N) if using
// m16n8k8.
template <typename MMA, typename Tensor>
DEVICE auto& convert_layout(const Tensor& acc) {
    auto acc_layout = acc.layout();

    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(cute::rank(acc_layout))::value == 3);

    constexpr int mma_shape_K = cute::get<2>(typename MMA::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);

    if constexpr (mma_shape_K == 8) {
        IndexedTensor_<decltype(acc)> indexed_tensor(acc);
        return indexed_tensor;
    } else {
        // (4, MMA_M, (2, MMA_N / 2)))
        auto l = cute::logical_divide(acc_layout, Shape<X, X, _2>{});
        auto new_layout = make_layout(make_layout(get<0>(l), get<2, 0>(l)),
                                      get<1>(l), get<2, 1>(l));
        auto new_tensor = make_tensor(acc.data(), new_layout);

        IndexedTensor_<decltype(new_tensor)> indexed_tensor(new_tensor);
        return indexed_tensor;
    }
};

template <const int m, const int n>
DEVICE auto& get_acc(const auto& tiled_mma) {
    auto acc = partition_fragment_C(tiled_mma, Shape<Int<m>, Int<n>>{});
    clear(acc);

    return acc;
}

template <typename TensorA, typename TensorB, typename TensorAcc>
DEVICE void gemm(const auto& mma, const TensorA& a, const TensorB& b,
                 TensorAcc& acc) {
    cute::gemm(mma, a, b, acc);  // compute
}

template <typename Layout>
DEVICE void DebugPrint(const cutlass::half_t* data) {
    const __half* ptr = reinterpret_cast<const __half*>(data);

    auto ids = Layout{};
    int row = num_rows<Layout>;
    int col = num_cols<Layout>;

    for (int i = 0; i < row; ++i) {
        printf("%d\t", i);
        for (int j = 0; j < col; ++j)
            printf("%.2f,", __half2float(ptr[ids(i, j)]));
        printf("\n");
    }
}

/**
 * @brief A struct that represents an asynchronous copy of TWO
 * operands operation from global memory to shared memory.
 *
 * This struct is used in the context of a matrix-matrix
 * multiplication kernel. It performs a tiled copy operation from a
 * source tensor in global memory to a destination tensor in shared
 * memory. The copy operation is performed in multiple stages, where
 * each stage copies a tile of the source tensor to the destination
 * tensor. The number of stages is specified by the template
 * parameter kNumStages.
 */
template <typename TiledCopy, typename SrcTensor1, typename DstTensor1,
          typename SrcTensor2, typename DstTensor2>
struct CopyAsyncG2S {
   public:
    CUTE_DEVICE
    CopyAsyncG2S(int num_stages, TiledCopy& tiled_copy, SrcTensor1& s1, int ss1,
                 DstTensor1& d1, int ds1, SrcTensor2& s2, int ss2,
                 DstTensor2& d2, int ds2)
        : num_stages(num_stages),
          tiled_copy(tiled_copy),
          src1(s1),
          src_stride1(ss1),
          dst1(d1),
          dst_stride1(ds1),
          src2(s2),
          src_stride2(ss2),
          dst2(d2),
          dst_stride2(ds2),
          iter(0) {}

    // All threads within a CTA work together to load the first data
    // tile from global memory to shared memory.
    CUTE_DEVICE
    void copy() {
        CUTE_UNROLL
        for (int i = 0; i < size<1>(src1); ++i) {
            CUTE_UNROLL
            for (int j = 0; j < size<2>(src1); ++j) {
                cute::copy(tiled_copy, src1(_, i, j), dst1(_, i, j));
            }
        }

        CUTE_UNROLL
        for (int i = 0; i < size<1>(src2); ++i) {
            CUTE_UNROLL
            for (int j = 0; j < size<2>(src2); ++j) {
                cute::copy(tiled_copy, src2(_, i, j), dst2(_, i, j));
            }
        }
        commit_copy_group();
        next();

        if ((iter + 1) % num_stages == 0) cycle_dst();

        ++iter;
    }

    /**
     * @param n1, for operand B, ON AVERAGE, how many cp.async
     operations need to be issued in a single compute iteration.
     * @param N1, for the operand A, how many cp.async operations
     are needed to issue IN TOTAL by a single thread.
     * @param stride1, for the operand A, how many cp.async
     operations are needed to issue in a single row.
     * @param n2, for operand B, ON AVERAGE, how many cp.async
     operations need to be issued in a single compute iteration.
     * @param N2, for the operand B, how many cp.async operations
     are needed to issue IN TOTAL by a single thread.
     * @param stride2, for the operand B, how many cp.async
     operations are needed to issue in a single row.
     */
    CUTE_DEVICE
    void copy2(int idx, int n1, int N1, int stride1, int n2, int N2,
               int stride2) {
        CUTE_UNROLL
        for (int i = 0; (i < n1) && (i + idx * n1 < N1); ++i) {
            int pos = i + idx * n1;
            int row = pos / stride1;
            int col = pos % stride1;
            cute::copy(tiled_copy, src1(_, row, col), dst1(_, row, col));
        }
        CUTE_UNROLL
        for (int i = 0; (i < n2) && (i + idx * n2 < N2); ++i) {
            int pos = i + idx * n2;
            int row = pos / stride2;
            int col = pos % stride2;
            cute::copy(tiled_copy, src2(_, row, col), dst2(_, row, col));
        }
    }

    template <int N>
    CUTE_DEVICE void wait_group() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
    }

    CUTE_DEVICE
    void commit_copy_group() { cute::cp_async_fence(); }

    CUTE_DEVICE
    void next() {
        src1.data() = src1.data() + src_stride1;
        src2.data() = src2.data() + src_stride2;

        dst1.data() = dst1.data() + dst_stride1;
        dst2.data() = dst2.data() + dst_stride2;
    }

    CUTE_DEVICE
    void cycle_dst() {
        dst1.data() = dst1.data() + (-dst_stride1 * num_stages);
        dst2.data() = dst2.data() + (-dst_stride2 * num_stages);
    }

   private:
    TiledCopy& tiled_copy;
    SrcTensor1& src1;
    const int src_stride1;
    DstTensor1& dst1;
    const int dst_stride1;
    SrcTensor2& src2;
    const int src_stride2;
    DstTensor2& dst2;
    const int dst_stride2;
    const int num_stages;
    int iter;
};

template <typename TiledCopy, typename SrcTensor, typename DstTensor>
CUTE_DEVICE void R2S_copy(TiledCopy& tiled_copy, SrcTensor& src, DstTensor& dst,
                          int tid) {
    auto copy_thrd = tiled_copy.get_thread_slice(threadIdx.x);
    auto src_copy_view = copy_thrd.retile_S(src);
    auto dst_thrd = copy_thrd.partition_D(dst);
    cute::copy(tiled_copy, src_copy_view, dst_thrd);
}

template <typename TiledCopy, typename SrcTensor, typename DstTensor>
CUTE_DEVICE void S2G_copy(TiledCopy& tiled_copy, SrcTensor& src, DstTensor& dst,
                          int tid) {
    auto copy_thrd = tiled_copy.get_thread_slice(tid);
    auto src_thrd = copy_thrd.partition_S(src);
    auto dst_thrd = copy_thrd.partition_D(dst);

    CUTE_UNROLL
    for (int i = 0; i < size<1>(dst_thrd); ++i) {
        CUTE_UNROLL
        for (int j = 0; j < size<2>(dst_thrd); ++j)
            cute::copy(tiled_copy, src_thrd(_, i, j), dst_thrd(_, i, j));
    }
}

}  // namespace cuda_kernel
}  // namespace core
}  // namespace kaleido
