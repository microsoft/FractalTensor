import torch

import triton
import triton.language as tl
import argparse

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 64,
                'BLOCK_SIZE_P': 32
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 64
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 64,
                'BLOCK_SIZE_P': 32
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 64
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 128
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'BLOCK_SIZE_P': 32
            },
            num_stages=2,
            num_warps=4),
    ],
    key=['M', 'N', 'K', 'P'],
)
@triton.jit
def backToBackGemm_kernel(
        a_ptr, b_ptr, c_ptr, d_ptr, M, K, N, P, stride_am, stride_ak,
        stride_bk, stride_bn, stride_cn, stride_cp, stride_dm, stride_dp,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_P: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(N, P),
        strides=(stride_cn, stride_cp),
        offsets=(0, pid_p * BLOCK_SIZE_P),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_P),
        order=(1, 0),
    )
    d = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_P], dtype=tl.float32)
    for start_n in range(0, N, BLOCK_SIZE_N):
        c = tl.load(c_block_ptr, boundary_check=(0, ), padding_option='zero')
        p = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        for start_k in range(0, K, BLOCK_SIZE_K):
            a = tl.load(
                a_block_ptr, boundary_check=(1, ), padding_option='zero')
            b = tl.load(
                b_block_ptr, boundary_check=(0, 1), padding_option='zero')
            p += tl.dot(a, b)
            # p = p.to(tl.float16)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

        p = p.to(tl.float16)
        d += tl.dot(p, c)
        a_block_ptr = tl.advance(a_block_ptr, (0, -K))
        b_block_ptr = tl.advance(b_block_ptr, (-K, BLOCK_SIZE_N))
        c_block_ptr = tl.advance(c_block_ptr, (BLOCK_SIZE_N, 0))

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_p = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    d_ptrs = d_ptr + offs_m[:, None] * stride_dm + offs_p[None, :] * stride_dp
    mask = (offs_m < M)[:, None] & (offs_p < P)[None, :]
    d = d.to(tl.float16)
    tl.store(d_ptrs, d, mask=mask)


def backToBackGemm(a, b, c):
    assert a.shape[1] == b.shape[0], "incompatible dimensions a-b"
    assert b.shape[1] == c.shape[0], "incompatible dimensions b-c"
    M, K = a.shape
    K, N = b.shape
    N, P = c.shape

    d = torch.empty((M, P), device=a.device, dtype=torch.float16)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']),
                triton.cdiv(P, META['BLOCK_SIZE_P']), 1)

    backToBackGemm_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        d_ptr=d,
        M=M,
        N=N,
        K=K,
        P=P,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cn=c.stride(0),
        stride_cp=c.stride(1),
        stride_dm=c.stride(0),
        stride_dp=c.stride(1))
    return d


def accept_test(a, b, c):
    triton_output = backToBackGemm(a, b, c)
    torch_output = torch.matmul(torch.matmul(a, b), c)
    print(f"Triton_output={triton_output}")
    print(f"Torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


def run_test(test_case, OUTPUT_FILE):
    torch.manual_seed(0)
    for case in test_case:
        M, K, N, P = case
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        c = torch.randn((N, P), device='cuda', dtype=torch.float16)
        # accept_test(a, b, c)
        ms = triton.testing.do_bench(
            lambda: backToBackGemm(a, b, c),
            warmup=25,
            rep=100,
            return_mode='mean')
        print(f"[{M}, {K}][{K}, {N}][{N}, {P}]\t" f"Baseline(ms): {ms}ms")
        if OUTPUT_FILE:
            with open(OUTPUT_FILE, 'a') as fout:
                fout.write(f"[{M}, {K}][{K}, {N}][{N}, {P}]\t"
                           f"Baseline(ms): {ms}ms\n")


def parse_test_args():
    parser = argparse.ArgumentParser(description='BacktoBack GEMMs')
    parser.add_argument(
        '--output_file', type=str, help='Output file path', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    test_case = [[8192, 64, 256, 64], [8192, 64, 512, 64],
                 [16384, 64, 256, 64], [16384, 64, 256, 64]]
    cmd_args = parse_test_args()
    OUTPUT_FILE = cmd_args.output_file
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'w') as fout:
            fout.write("GEMM Shape\tTriton(ms)\n")
    torch.manual_seed(0)
    run_test(test_case, OUTPUT_FILE)
