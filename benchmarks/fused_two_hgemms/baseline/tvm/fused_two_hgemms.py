import logging
import sys
import warnings

import torch
import numpy as np
import time
import math
import argparse

from pathlib import Path
import tvm
from tvm import te
from tvm import testing
from tvm import autotvm
from tvm.target import Target
from tvm import auto_scheduler
from tvm import topi
from tvm.autotvm.tuner import XGBTuner


def tvm_solver(parameter,
               target,
               dtype,
               use_logFile,
               logFile="backToBackGemm.json"):
    @auto_scheduler.register_workload
    def backToBackGemm_kernel(M, K, N, P, dtype):
        A = te.placeholder((M, K), name="A", dtype=dtype)
        B = te.placeholder((K, N), name="B", dtype=dtype)
        C = te.placeholder((N, P), name="C", dtype=dtype)

        k = te.reduce_axis((0, K), name="L-K")
        n = te.reduce_axis((0, N), name="L-N")
        P_ = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            name="matmul",
            attrs={"layout_free_placeholders": [B]},
        )
        D = te.compute(
            (M, P),
            lambda i, j: te.sum(P_[i, n] * C[n, j], axis=n),
            name="matmul",
            attrs={"layout_free_placeholders": [C]},
        )
        return [A, B, C, D]

    M, K, N, P = parameter
    logFile = f"backToBackGemm_{M}_{K}_{N}_{P}.json"

    tasks = [
        tvm.auto_scheduler.SearchTask(
            func=backToBackGemm_kernel,
            args=(M, K, N, P, dtype),
            target=target),
    ]

    tuning_rounds = 1000
    tuner = auto_scheduler.TaskScheduler(tasks)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=tuning_rounds * len(tasks),
        measure_callbacks=[auto_scheduler.RecordToFile(logFile)],
        verbose=2,
    )

    if use_logFile is False:
        autosch_time_start = time.time()

        tuner.tune(tune_option)

        autosch_time_end = time.time()
        print("auto scheduler cost", (autosch_time_end - autosch_time_start),
              "s")

    backToBackGemm_kernel_sch, backToBackGemm_kernel_args = tasks[
        0].apply_best(logFile)
    backToBackGemm_mod = tvm.lower(
        backToBackGemm_kernel_sch,
        backToBackGemm_kernel_args,
        simple_mode=True)
    backToBackGemm = tvm.build(backToBackGemm_kernel_sch,
                               backToBackGemm_kernel_sch, target)

    return backToBackGemm


def run_test(test_case):
    torch.manual_seed(0)
    warmup = 25
    iter = 100
    for case in test_case:
        dtype = "float16"
        dev = tvm.cuda()
        M, K, N, P = case

        a = np.zeros([M, K], dtype=dtype)
        b = np.zeros([K, N], dtype=dtype)
        c = np.zeros([N, P], dtype=dtype)
        d = (np.matmul(a, b), c)

        a_tvm = tvm.nd.array(np.zeros([M, K], dtype=dtype), dev)
        b_tvm = tvm.nd.array(np.zeros([K, N], dtype=dtype), dev)
        c_tvm = tvm.nd.array(np.zeros([N, P], dtype=dtype), dev)
        d_tvm = tvm.nd.array(np.zeros([M, P], dtype=dtype), dev)

        parameter = [M, K, N, P]
        use_logFile = False
        target = "cuda -libs=cublas"
        backToBackGemm = \
                tvm_solver(parameter, target, dtype, use_logFile)
        backToBackGemm(a_tvm, b_tvm, c_tvm, d_tvm)

        for _ in range(warmup):
            backToBackGemm(a_tvm, b_tvm, c_tvm, d_tvm)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        total_time = 0
        for _ in range(iter):
            torch.cuda.synchronize()
            start_event.record()
            backToBackGemm(a_tvm, b_tvm, c_tvm, d_tvm)
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

        # print(f"TVM_output={d}")
        # print(f"NumPy_output={d_tvm}")
        # if torch.allclose(d, d_tvm, atol=1e-2, rtol=0):
        #     print("✅ TVM and NumPy match")
        # else:
        #     print("❌ TVM and NumPy differ")


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
            fout.write("GEMM Shape\tTVM(ms)\n")
    run_test(test_case)
