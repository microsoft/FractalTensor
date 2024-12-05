from asyncio import gather
import numpy as np
import argparse
import types


def tvm_bigbird(batch_size, hidden_size, len, block_size, window_size,
                random_size, global_size, OUTPUT_FILE):
    from tvm import relay
    from tvm.relay import testing
    import tvm
    from tvm import te
    from tvm.contrib import graph_executor
    import tvm.testing
    import math

    import tvm.auto_scheduler as auto_scheduler
    from tvm.autotvm.tuner import XGBTuner
    from tvm import topi, autotvm
    import logging
    from datetime import datetime
    import sys
    import argparse
    # Enable debug logs
    import logging
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    target_name = 'cuda -libs=cublas'
    target = tvm.target.Target(target_name)
    dtype, itype = 'float32', 'int32'

    # udf0
    # qk
    @auto_scheduler.register_workload
    def bigbird_qk_global_row(N, L, B, H, W, G, dtype):
        X = te.placeholder((N, L, B, H), name='X', dtype=dtype)
        Y = te.placeholder(
            (N, L + W // 2 + W // 2, B, H), name='Y', dtype=dtype)
        k = te.reduce_axis((0, H), name='k')
        out_shape = (N, 2 * G, B, B * L)

        def qk_mm(n, l, i, j):
            return te.sum(
                X[n, tvm.tir.if_then_else(l < G, l, L - G + l), i, k] *
                Y[n, j // B + W // 2, j % B, k],
                axis=k)

        R = te.compute(out_shape, qk_mm, name='R')

        return [X, Y, R]

    # softmax
    @auto_scheduler.register_workload
    def global_row_softmax(N, L, B, G, dtype):
        x = te.placeholder((N, 2 * G, B, B * L))
        out = topi.nn.softmax(x) / (math.sqrt(hidden_size))

        return [x, out]

    # wv
    @auto_scheduler.register_workload
    def bigbird_wv_global_row(N, L, B, H, W, G, dtype):
        X = te.placeholder((N, 2 * G, B, B * L), name='X', dtype=dtype)
        Y = te.placeholder(
            (N, L + W // 2 + W // 2, B, H), name='Y', dtype=dtype)
        k = te.reduce_axis((0, B * L), name='k')
        out_shape = (N, 2 * G, B, H)

        def wv_mm(n, l, i, j):
            return te.sum(
                X[n, l, i, k] * Y[n, W // 2 + k // B, k % B, j], axis=k)

        R = te.compute(
            out_shape,
            wv_mm,
            name='R',
        )

        return [X, Y, R]

    # udf1
    @auto_scheduler.register_workload
    def bigbird_qk_SDDMM_global_window(N, L, B, H, W, R, G, dtype):
        X = te.placeholder((N, L, B, H), name='X', dtype=dtype)
        Y = te.placeholder(
            (N, L + W // 2 + W // 2, B, H), name='Y', dtype=dtype)
        k = te.reduce_axis((0, H), name='k')
        out_shape = (N, L - 2 * G, B, B * (W + 2))

        def qk_sddmm(n, l, i, j):
            return te.sum(
                X[n, l + G, i, k] *
                Y[n,
                  tvm.tir.if_then_else(
                      j < G * B, W // 2,
                      tvm.tir.if_then_else(j < 2 * G * B, L - 1, l + j // B -
                                           2)), j % B, k],
                axis=k)

        Res = te.compute(
            out_shape,
            qk_sddmm,
            name='Res',
        )

        return [X, Y, Res]

    @auto_scheduler.register_workload
    def bigbird_qk_SDDMM_random(N, L, B, H, W, R, G, dtype):
        X = te.placeholder((N, L, B, H), name='X', dtype=dtype)
        GatheredK = te.placeholder((N, R, B, H), name='GatheredK', dtype=dtype)
        k = te.reduce_axis((0, H), name='k')
        out_shape = (N, L - 2 * G, B, B * R)

        def qk_sddmm(n, l, i, j):
            return te.sum(
                X[n, l + G, i, k] * GatheredK[n, j // B, j % B, k], axis=k)

        Res = te.compute(
            out_shape,
            qk_sddmm,
            name='Res',
        )

        return [X, GatheredK, Res]

    @auto_scheduler.register_workload
    def sparse_row_concat_softmax(N, L, B, H, W, R, G, dtype):
        X = te.placeholder(
            (N, L - 2 * G, B, B * (W + 2)), name='X', dtype=dtype)
        Y = te.placeholder((N, L - 2 * G, B, B * R), name='Y', dtype=dtype)
        concat_shape = (N, L - 2 * G, B, B * (W + R + 2))
        weight = te.compute(
            concat_shape,
            lambda i, j, k, l: tvm.tir.if_then_else(
                l < B * (W + 2), X[i, j, k, l], Y[i, j, k, l - B * (W + 2)])
        )
        out = topi.nn.softmax(weight) / (math.sqrt(hidden_size))

        return [X, Y, out]

    @auto_scheduler.register_workload
    def sparse_row_softmax(N, L, B, H, W, R, G, dtype):
        x = te.placeholder((N, L - 2 * G, B, B * (W + R + 2)))
        out = topi.nn.softmax(x) / (math.sqrt(hidden_size))

        return [x, out]

    # N = 1, L = 4096, B = 1, H = 64, W = 1, R = 1, G = 1

    @auto_scheduler.register_workload
    def bigbird_wv_SPMM_global_window(N, L, B, H, W, R, G, dtype):
        X = te.placeholder(
            (N, L - 2 * G, B, B * (W + R + 2)), name='X', dtype=dtype)
        Y = te.placeholder(
            (N, L + W // 2 + W // 2, B, H), name='Y', dtype=dtype)
        k = te.reduce_axis((0, B * (W + 2 * G)), name='k1')
        out_shape = (N, L - 2 * G, B, H)
        # (1, 4094, 1, 64)
        res = te.compute(out_shape, lambda n, l, i, j: te.sum(
            X[n, l, i, k]*Y[
                n,
                tvm.tir.if_then_else(
                    k < G*B, W//2, tvm.tir.if_then_else(k < 2*G*B, L-1+W//2, l+k//B-2)),
                k % B,
                j],
            axis=k),
            name='res')

        return [X, Y, res]

    @auto_scheduler.register_workload
    def bigbird_wv_SPMM_random(N, L, B, H, W, R, G, dtype):
        X = te.placeholder(
            (N, L - 2 * G, B, B * (W + R + 2)), name='X', dtype=dtype)
        GatheredV = te.placeholder((N, R, B, H), name='GatheredV', dtype=dtype)
        k = te.reduce_axis((0, B * R), name='k')
        out_shape = (N, L - 2 * G, B, H)

        res = te.compute(
            out_shape, lambda n, l, i, j: te.sum(
                X[n, l, i, k+B*(W+2)] * GatheredV[n, k//B, k % B, j], axis=k),
            name='res')

        return [X, GatheredV, res]

    @auto_scheduler.register_workload
    def bigbird_wv_SPMM_reduce(N, L, B, H, W, R, G, dtype):
        out_shape = (N, L - 2 * G, B, H)
        X = te.placeholder(out_shape, name='X', dtype=dtype)
        Y = te.placeholder(out_shape, name='Y', dtype=dtype)

        res = te.compute(
            out_shape,
            lambda n, l, i, j: X[n, l, i, j] + Y[n, l, i, j],
            name='res')

        return [X, Y, res]

    args = (batch_size, len // block_size, block_size, hidden_size,
            window_size, random_size, global_size, dtype)

    tasks = [
        tvm.auto_scheduler.SearchTask(
            func=bigbird_qk_global_row,
            args=(batch_size, len // block_size, block_size, hidden_size,
                  window_size, global_size, dtype),
            target=target),
        tvm.auto_scheduler.SearchTask(
            func=global_row_softmax,
            args=(batch_size, len // block_size, block_size, global_size,
                  dtype),
            target=target),
        tvm.auto_scheduler.SearchTask(
            func=bigbird_wv_global_row,
            args=(batch_size, len // block_size, block_size, hidden_size,
                  window_size, global_size, dtype),
            target=target),
        tvm.auto_scheduler.SearchTask(
            func=bigbird_qk_SDDMM_global_window, args=args, target=target),
        tvm.auto_scheduler.SearchTask(
            func=bigbird_qk_SDDMM_random, args=args, target=target),
        tvm.auto_scheduler.SearchTask(
            func=sparse_row_concat_softmax, args=args, target=target),
        tvm.auto_scheduler.SearchTask(
            func=bigbird_wv_SPMM_global_window, args=args, target=target),
        tvm.auto_scheduler.SearchTask(
            func=bigbird_wv_SPMM_random, args=args, target=target),
        tvm.auto_scheduler.SearchTask(
            func=bigbird_wv_SPMM_reduce, args=args, target=target),
    ]

    shape = (batch_size, len, block_size, hidden_size, window_size,
             global_size)

    log_file = f'ansor.{shape}.json'

    tuner = auto_scheduler.TaskScheduler(tasks)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    tuner.tune(tune_option)

    funcs = []

    for task in tasks:
        sch, args = task.apply_best(log_file)
        funcs.append(tvm.build(sch, args, target))

    dev = tvm.cuda()

    Q = np.ones(
        [batch_size, len // block_size, block_size, hidden_size],
        dtype="float32")
    K = np.ones(
        [
            batch_size,
            len // block_size + window_size // 2 + window_size // 2,
            block_size, hidden_size
        ],
        dtype="float32")
    V = np.ones(
        [
            batch_size,
            len // block_size + window_size // 2 + window_size // 2,
            block_size, hidden_size
        ],
        dtype="float32")

    GatheredK = np.ones(
        [batch_size, random_size, block_size, hidden_size], dtype="float32")
    GatheredV = np.ones(
        [batch_size, random_size, block_size, hidden_size], dtype="float32")

    Q_tvm = tvm.nd.array(Q, device=dev)
    K_tvm = tvm.nd.array(K, device=dev)
    V_tvm = tvm.nd.array(V, device=dev)
    GatheredK_tvm = tvm.nd.array(GatheredK, device=dev)
    GatheredV_tvm = tvm.nd.array(GatheredV, device=dev)

    global_row_weight = tvm.nd.empty(
        (batch_size, 2 * global_size, block_size, len), device=dev)
    global_row_weight_softmax = tvm.nd.empty(
        (batch_size, 2 * global_size, block_size, len), device=dev)
    global_row_res = tvm.nd.empty(
        (batch_size, 2 * global_size, block_size, hidden_size), device=dev)

    global_window_weight = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         (2 + window_size) * block_size),
        device=dev)
    random_weight = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         random_size * block_size),
        device=dev)
    sparse_row_weight_softmax = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         (2 + window_size + random_size) * block_size),
        device=dev)
    sparse_row_res = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         hidden_size),
        device=dev)
    sparse_row_res1 = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         hidden_size),
        device=dev)
    sparse_row_res2 = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         hidden_size),
        device=dev)

    inputs_funcs = [
        # --------------------Global Row--------------------
        (Q_tvm, K_tvm, global_row_weight),  # bigbird_qk_global_row
        (global_row_weight, global_row_weight_softmax),  # global_row_softmax
        (global_row_weight_softmax, V_tvm,
         global_row_res),  # bigbird_wv_global_row

        # --------------------Sparse Row--------------------
        (Q_tvm, K_tvm, global_window_weight),  # bigbird_qk_SDDMM_global_window
        (Q_tvm, GatheredK_tvm, random_weight),  # bigbird_qk_SDDMM_random
        # sparse_row_concat_softmax
        (global_window_weight, random_weight, sparse_row_weight_softmax),
        # bigbird_wv_SPMM_global_window
        (sparse_row_weight_softmax, V_tvm, sparse_row_res),
        (sparse_row_weight_softmax, GatheredV_tvm,
         sparse_row_res1),  # bigbird_wv_SPMM_random
        (sparse_row_res, sparse_row_res1, sparse_row_res2)
    ]

    for func, inputs in zip(funcs, inputs_funcs):
        func(*inputs)

    warmup_num = 5
    test_num = 10
    time_log = []

    for func, inputs in zip(funcs, inputs_funcs):
        evaluator = func.time_evaluator(
            func.entry_name, dev, number=warmup_num)
        evaluator(*inputs)
        evaluator = func.time_evaluator(func.entry_name, dev, number=test_num)
        time_ms = np.median(evaluator(*inputs).results) * 1000
        time_log.append(time_ms)

    blocks = batch_size * (len // block_size - 2 * global_size) * (
        window_size + random_size +
        2 * global_size) + batch_size * 2 * global_size * len // block_size

    operation_per_block = 4 * block_size * block_size * \
        hidden_size + 2 * block_size * block_size

    operations = blocks * operation_per_block

    operations = operations >> 25
    GFLOPs = operations / (sum(time_log) / 1000 * 32)

    file_name = f"bigbird_tvm_data_{batch_size}_{hidden_size}_{block_size}"
    with open(file_name, 'a', encoding='utf-8') as f:
        f.writelines(f"{batch_size}_{hidden_size}_{block_size}\n:")
        f.writelines(f'Time breakdown (ms):, {time_log}\n')
        f.writelines("Average e2e time: %.3f ms\n" % (sum(time_log)))
        f.writelines(f"GFLOPs:{GFLOPs}\n")

    return sum(time_log), GFLOPs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v in ('True'):
        return True
    elif v in ('False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_test_args():
    parser = argparse.ArgumentParser(description='Bigbird')
    parser.add_argument(
        '--seq_len', type=int, help='Sequence length', default=4096)
    parser.add_argument(
        '--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument(
        '--hidden_size', type=int, help='Hidden size', default=512)
    parser.add_argument(
        '--block_size', type=int, help='Block size', default=64)
    parser.add_argument(
        '--output_file', type=str, help='Output file path', default=None)
    parser.add_argument(
        '--default_test',
        type=str2bool,
        help='Whether to run the default test',
        default=False)
    return parser.parse_args()


def output_file(OUTPUT_FILE, cmd_args, run_time):
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'a') as fout:
            fout.write(
                f"{cmd_args.batch_size}\t{cmd_args.seq_len}\t{cmd_args.hidden_size}\t{cmd_args.block_size}\t"
                f"{run_time}\n")


if __name__ == '__main__':
    global_size = 1
    window_size = 3
    random_size = 3

    cmd_args = parse_test_args()
    DEFAULT_TEST = cmd_args.default_test
    OUTPUT_FILE = cmd_args.output_file
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'w') as fout:
            fout.write(
                "batch size\t sequence length\thidden\tblock size\telapsed time(ms)\n"
            )

    if DEFAULT_TEST:
        test1_cmd_args = types.SimpleNamespace(
            seq_len=4096, batch_size=32, hidden_size=512, block_size=64)
        run_time1, _ = tvm_bigbird(32, 512, 4096, 64, window_size, random_size,
                                   global_size, OUTPUT_FILE)
        output_file(OUTPUT_FILE, test1_cmd_args, run_time1)
        test2_cmd_args = types.SimpleNamespace(
            seq_len=8192, batch_size=32, hidden_size=512, block_size=64)
        run_time2, _ = tvm_bigbird(32, 512, 8192, 64, window_size, random_size,
                                   global_size, OUTPUT_FILE)
        output_file(OUTPUT_FILE, test2_cmd_args, run_time2)
    else:
        len = cmd_args.seq_len
        batch_size = cmd_args.batch_size
        hidden_size = cmd_args.hidden_size
        block_size = cmd_args.block_size
        run_time, _ = tvm_bigbird(batch_size, hidden_size, len, block_size,
                                  window_size, random_size, global_size,
                                  OUTPUT_FILE)
        output_file(OUTPUT_FILE, cmd_args, run_time)
