import os
import time
import numpy as np
import argparse
import jax
import torch
from jax import jit
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true ')


@jit
def backToBackGemm(a, b, c):
    d = jnp.dot(jnp.dot(a, b), c)
    return d


@jit
def backToBackGemm2(a, b, c):
    d = jnp.linalg.multi_dot([a, b, c])
    return d


def accept_test(M, K, N, P):
    a = np.random.normal(size=(M, K)).astype(np.float16)
    b = np.random.normal(size=(K, N)).astype(np.float16)
    c = np.random.normal(size=(N, P)).astype(np.float16)
    d = np.dot(np.dot(a, b), c)

    a_j = jax.device_put(a)
    b_j = jax.device_put(b)
    c_j = jax.device_put(c)
    d_j = backToBackGemm(a_j, b_j, c_j)

    print(f"NumPy_output={d}")
    print(f"JAX_output={d_j}")
    if np.allclose(d, d_j, atol=1e-1, rtol=0):
        print("✅ JAX and NumPy match")
    else:
        print("❌ JAX and NumPy differ")


def run_test(test_case):
    warmup = 25
    iter = 100
    for case in test_case:
        M, K, N, P = case
        a = jax.device_put(np.random.normal(size=(M, K)).astype(np.float16))
        b = jax.device_put(np.random.normal(size=(K, N)).astype(np.float16))
        c = jax.device_put(np.random.normal(size=(N, P)).astype(np.float16))
        # quantiles = [0.5, 0.2, 0.8]

        for _ in range(warmup):
            backToBackGemm(a, b, c)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        total_time = 0
        for _ in range(iter):
            torch.cuda.synchronize()
            start_event.record()
            backToBackGemm(a, b, c)
            torch.cuda.synchronize()
            end_event.record()
            elapsed = start_event.elapsed_time(end_event)
            total_time += elapsed

        print(f"[{M}, {K}][{K}, {N}][{N}, {P}]\t"
              f"Baseline(ms): {total_time * 1000 / iter}ms")
        if OUTPUT_FILE:
            with open(OUTPUT_FILE, 'a') as fout:
                fout.write(f"[{M}, {K}][{K}, {N}][{N}, {P}]\t"
                           f"Baseline(ms): {total_time * 1000 / iter}ms\n")


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
            fout.write("GEMM Shape\tJAX(ms)\n")
    run_test(test_case)

    # Print the HLO
    # xla_comp = jax.xla_computation(backToBackGemm)(a, b, c)
    # print(xla_comp.as_hlo_text())
