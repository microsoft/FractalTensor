#include "util/cuda_timer.hpp"
#include "utils/util.hpp"

template <typename InType,
          typename AccType,                                      //
          typename GIteratorA, typename SharedA, typename RegA,  //
          typename SharedALoader, typename RegALoader,           //
          typename GIteratorB, typename SharedB, typename RegB,  //
          typename SharedBLoader, typename RegBLoader,           //
          typename GIteratorC, typename SharedC, typename RegC,  //
          typename SharedCLoader, typename RegCLoader,           //
          typename RegAcc, typename RegAccCast, typename GlobalD, typename RegD,
          typename RegDCast, typename DStorer, typename ConvertAcc,
          typename ConvertO, typename RegVec, typename CopyVec, typename RowMax,
          typename BroadcastSub, typename BroadcastMul, typename BroadcastDiv,
          typename BlockExp, typename BlockAdd, typename VecMax,
          typename VecAdd, typename VecSub, typename VecMul, typename VecExp>
__global__ void KeFlashAttention(const InType* dA, const InType* dB,
                                 const InType* dC, InType* dD, int kM, int kN,
                                 int kK, int kP, int kTM, int kTN, int kTK,
                                 int kTP) {
    // Advance to the global data tile to the current CTA.
    const InType* A = dA + blockIdx.z * (kM * kK) + blockIdx.x * (kTM * kK);
    const InType* B = dB + blockIdx.z * (kK * kN);
    const InType* gC_ptr =
        dC + blockIdx.z * (kN * kP) + blockIdx.y * (kTP * kN);

    InType* gD_ptr = dD + blockIdx.z * (kM * kP) + blockIdx.x * (kTM * kP) +
                     (blockIdx.y * kTP);

    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto* shm = reinterpret_cast<InType*>(shared_buf);

    InType* sA_ptr = shm;
    InType* sB_ptr = shm + SharedA::kNumel;
    InType* sC_ptr = shm + SharedA::kNumel + SharedB::kNumel;

    GIteratorA gAs(A);
    SharedA sA(sA_ptr);
    RegA rA;

    SharedALoader load_sa;
    RegALoader load_ra;

    GIteratorB gBs(B);
    SharedB sB(sB_ptr);
    RegB rB;

    SharedBLoader load_sb;
    RegBLoader load_rb;

    GIteratorC gCs(gC_ptr);
    SharedC sC(sC_ptr);

    SharedCLoader load_sc;
    RegCLoader load_rc;
    RegC rC;

    // RegD rD_f32;
    RegD unnormized_attn_block_f32;

    RegDCast rD;
    RegDCast unnormized_attn_block;

    RegAcc attn_block_f32;
    RegAccCast attn_block;

    RegVec prev_norm_vec;
    RegVec cur_norm_vec;

    RegVec prev_max_vec;
    RegVec cur_max_vec;
    RegVec new_max_vec;

    RegVec prev_sum_vec;
    RegVec cur_sum_vec;
    RegVec new_sum_vec;

    RowMax row_max;
    CopyVec copy_vec;

    BroadcastSub broadcast_sub;
    BroadcastMul broadcast_mul;

    BlockExp block_exp;
    BlockAdd block_add;

    VecMax vec_max;
    VecAdd vec_add;
    VecSub vec_sub;
    VecMul vec_mul;
    VecExp vec_exp;

    ConvertAcc cast_acc;  // Convert acc to half precision
    ConvertO cast_o;      // Convert half precision to float.

    for (int n = 0; n < GIteratorC::sc0; ++n) {
        load_sc(gCs(n), sC);

        for (int k = 0; k < GIteratorA::sc1; ++k) {
            load_sa(gAs(k), sA);
            load_sb(gBs(k, n), sB);
            __copy_async();
            __syncthreads();

            load_ra(sA, rA);
            load_rb(sB, rB);
            __syncthreads();

            compute::gemm_(rA, rB, attn_block_f32);
        }
        load_rc(sC, rC);
        __syncthreads();

        cast_acc(attn_block_f32, attn_block);

        // Copy `cur_max_vec`, `cur_norm_vec` into `prev_max_vec`,
        // `prev_norm_vec`.
        copy_vec(cur_max_vec, prev_max_vec);
        copy_vec(cur_norm_vec, prev_norm_vec);

        // Compute row max.
        row_max(attn_block, cur_max_vec);

        // Broadcast subtract from `attn_block`.
        broadcast_sub(cur_max_vec, attn_block);

        // Compute exp in `attn_block`.
        block_exp(attn_block, attn_block);

        // Compute new max vector.
        vec_max(cur_max_vec, prev_max_vec, new_max_vec);

        // Renormalization for the previous block.
        vec_sub(prev_max_vec, new_max_vec, prev_norm_vec);
        vec_exp(prev_norm_vec, prev_norm_vec);

        // Renormalization for the current block.
        vec_sub(cur_max_vec, new_max_vec, cur_norm_vec);
        vec_exp(cur_norm_vec, cur_norm_vec);

        // Update normalization factor l(x)
        vec_mul(prev_norm_vec, prev_sum_vec, prev_sum_vec);
        vec_mul(cur_norm_vec, cur_sum_vec, cur_sum_vec);
        vec_add(prev_sum_vec, cur_sum_vec, new_sum_vec);

        // Compute unnormized attention block.
        compute::gemm_(attn_block, rC, unnormized_attn_block_f32);

        __syncthreads();

        cast_o(unnormized_attn_block_f32, unnormized_attn_block);

        vec_mul(prev_sum_vec, prev_norm_vec, prev_norm_vec);
        broadcast_mul(prev_norm_vec, rD);

        broadcast_mul(cur_norm_vec, unnormized_attn_block);

        block_add(rD, unnormized_attn_block, rD);

        // Cear the accumulator.
        attn_block_f32.clear();

        // Update max vector and sum vector.
        copy_vec(new_max_vec, prev_max_vec);
        copy_vec(new_sum_vec, prev_sum_vec);
    }
    __syncthreads();

    GlobalD gD(gD_ptr);
    DStorer storer_d;  // Store D tile from register to global.
    storer_d(rD, gD);
}

template <typename WholeShape, typename CtaTileShape, const int kBatch>
void run(bool check = true) {
    using InType = __half;
    using AccType = float;

    static constexpr int kM = dim_size<0, WholeShape>;
    static constexpr int kN = dim_size<1, WholeShape>;
    static constexpr int kK = dim_size<2, WholeShape>;
    static constexpr int kP = dim_size<3, WholeShape>;

    static constexpr int kTM = dim_size<0, CtaTileShape>;
    static constexpr int kTN = dim_size<1, CtaTileShape>;
    static constexpr int kTK = dim_size<2, CtaTileShape>;
    static constexpr int kTP = dim_size<3, CtaTileShape>;

    static_assert(kK == kTK,
                  "The current implementation requires kTK == K for now.");
    static_assert(kP == kTP,
                  "The current implementation requires kTP == P for now.");

    // initialize data
    thrust::host_vector<InType> h_a(kM * kK * kBatch);

    for (int i = 0; i < h_a.size(); ++i)
        h_a[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_b(kK * kN * kBatch);
    for (int i = 0; i < h_b.size(); ++i)
        h_b[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_c(kN * kP * kBatch);
    for (int i = 0; i < h_c.size(); ++i)
        h_c[i] = static_cast<InType>(rand_float());

    thrust::host_vector<InType> h_d(kM * kP * kBatch);
    thrust::fill(h_d.begin(), h_d.end(), 0.);

    thrust::device_vector<InType> d_a = h_a;
    thrust::device_vector<InType> d_b = h_b;
    thrust::device_vector<InType> d_c = h_c;
    thrust::device_vector<InType> d_d = h_d;

    const InType* A = thrust::raw_pointer_cast(d_a.data());
    const InType* B = thrust::raw_pointer_cast(d_b.data());
    const InType* C = thrust::raw_pointer_cast(d_c.data());
    InType* D = thrust::raw_pointer_cast(d_d.data());

    using Config =
        FlashAttentionTraits<InType, AccType, WholeShape, CtaTileShape>;

    using RegA = typename Config::RegA;
    using RegB = typename Config::RegB;
    using RegC = typename Config::RegC;
    using RegD = typename Config::RegD;
    using RegDCast = typename Config::RegDCast;
    using RegAcc = typename Config::RegAcc;
    using RegAccCast = typename Config::RegAccCast;

    using GIteratorA = typename Config::GIteratorA;
    using SharedA = typename Config::SharedA;
    using SharedALoader = typename Config::SharedALoader;
    using RegALoader = typename Config::RegALoader;

    using GIteratorB = typename Config::GIteratorB;
    using SharedB = typename Config::SharedB;
    using SharedBLoader = typename Config::SharedBLoader;
    using RegBLoader = typename Config::RegBLoader;

    using GIteratorC = typename Config::GIteratorC;
    using SharedC = typename Config::SharedC;
    using SharedCLoader = typename Config::SharedCLoader;
    using RegCLoader = typename Config::RegCLoader;

    using DStorer = typename Config::DStorer;

    using ConvertAcc = typename Config::ConvertHalf;
    using ConvertO = typename Config::ConvertO;

    using RegVec = typename Config::RegVec;

    using CopyVec = typename Config::CopyVec;
    using RowMax = typename Config::RowMax;

    using BroadcastSub = typename Config::BroadcastSub;
    using BroadcastMul = typename Config::BroadcastMul;
    using BroadcastDiv = typename Config::BroadcastDiv;

    using BlockExp = typename Config::BlockExp;
    using BlockAdd = typename Config::BlockAdd;

    using VecMax = typename Config::VecMax;
    using VecAdd = typename Config::VecAdd;
    using VecSub = typename Config::VecSub;
    using VecMul = typename Config::VecMul;
    using VecExp = typename Config::VecExp;

    int block_x = CeilDiv<kM, kTM>;
    int block_y = CeilDiv<kP, kTP>;
    int block_z = kBatch;

    dim3 grid(block_x, block_y, block_z);
    dim3 block(Config::kThreads, 1, 1);

    int shm_input = (kTM * kTK + kTK * kTN + kTN * kTP);
    int shm_output = kTM * kTP;
    int shm_size = shm_input < shm_output ? shm_output * sizeof(InType)
                                          : shm_input * sizeof(InType);

    auto kernel =
        &KeFlashAttention<InType, AccType,            //
                          GIteratorA, SharedA, RegA,  //
                          SharedALoader, RegALoader,  //
                          GIteratorB, SharedB, RegB,  //
                          SharedBLoader, RegBLoader,  //
                          GIteratorC, SharedC, RegC,  //
                          SharedCLoader, RegCLoader,  //
                          RegAcc, RegAccCast, typename Config::GlobalD, RegD,
                          RegDCast, DStorer, ConvertAcc, ConvertO, RegVec,
                          CopyVec, RowMax, BroadcastSub, BroadcastMul,
                          BroadcastDiv, BlockExp, BlockAdd, VecMax, VecAdd,
                          VecSub, VecMul, VecExp>;

    if (shm_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    }

    kernel<<<grid, block, shm_size, 0>>>(A, B, C, D, kM, kN, kK, kP, kTM, kTN,
                                         kTK, kTP);
}

int main() {
    // hidden 128
    run<FlashAttentionShape<128, 128, 128, 128>,
        FlashAttentionShape<64, 128, 128, 128>, 64>();

    run<FlashAttentionShape<256, 256, 128, 128>,
        FlashAttentionShape<64, 128, 128, 128>, 64>();

    run<FlashAttentionShape<512, 512, 128, 128>,
        FlashAttentionShape<64, 128, 128, 128>, 64>();

    run<FlashAttentionShape<768, 768, 128, 128>,
        FlashAttentionShape<64, 128, 128, 128>, 64>();

    run<FlashAttentionShape<1024, 1024, 128, 128>,
        FlashAttentionShape<64, 128, 128, 128>, 64>();

    run<FlashAttentionShape<1536, 1536, 128, 128>,
        FlashAttentionShape<64, 128, 128, 128>, 64>();

    run<FlashAttentionShape<2048, 2048, 128, 128>,
        FlashAttentionShape<64, 128, 128, 128>, 64>();

    // hidden 256
    run<FlashAttentionShape<128, 128, 256, 256>,
        FlashAttentionShape<64, 128, 256, 256>, 64>();

    run<FlashAttentionShape<256, 256, 256, 256>,
        FlashAttentionShape<64, 128, 256, 256>, 64>();

    run<FlashAttentionShape<512, 512, 256, 256>,
        FlashAttentionShape<64, 128, 256, 256>, 64>();

    run<FlashAttentionShape<768, 768, 256, 256>,
        FlashAttentionShape<64, 128, 256, 256>, 64>();

    run<FlashAttentionShape<1024, 1024, 256, 256>,
        FlashAttentionShape<64, 128, 256, 256>, 64>();

    run<FlashAttentionShape<1536, 1536, 256, 256>,
        FlashAttentionShape<64, 64, 256, 256>, 64>();

    run<FlashAttentionShape<2048, 2048, 256, 256>,
        FlashAttentionShape<64, 64, 256, 256>, 64>();

    return 0;
}