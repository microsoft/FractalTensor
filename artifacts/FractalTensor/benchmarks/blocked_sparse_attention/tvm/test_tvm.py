from asyncio import gather
import numpy as np


def tvm_bigbird(batch_size, hidden_size, len, block_size, window_size,
                random_size, global_size):
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
    def bigbird_qk_SDDMM_global(N, L, B, H, W, R, G, dtype):
        X = te.placeholder((N, L, B, H), name='X', dtype=dtype)
        Y = te.placeholder(
            (N, L + W // 2 + W // 2, B, H), name='Y', dtype=dtype)
        k = te.reduce_axis((0, H), name='k')
        out_shape = (N, L - 2 * G, B, 2 * B)

        def qk_sddmm(n, l, i, j):
            return te.sum(
                X[n, l + G, i, k] *
                Y[n,
                  tvm.tir.if_then_else(j < G * B, W // 2, L - 1), j % B, k],
                axis=k)

        Res = te.compute(
            out_shape,
            qk_sddmm,
            name='Res',
        )

        return [X, Y, Res]

    @auto_scheduler.register_workload
    def bigbird_qk_SDDMM_window(N, L, B, H, W, R, G, dtype):
        X = te.placeholder((N, L, B, H), name='X', dtype=dtype)
        Y = te.placeholder(
            (N, L + W // 2 + W // 2, B, H), name='Y', dtype=dtype)
        k = te.reduce_axis((0, H), name='k')
        out_shape = (N, L - 2 * G, B, W * B)

        def qk_sddmm(n, l, i, j):
            return te.sum(
                X[n, l + G, i, k] * Y[n, l + j // B - 2, j % B, k], axis=k)

        Res = te.compute(
            out_shape,
            qk_sddmm,
            name='Res',
        )

        return [X, Y, Res]

    @auto_scheduler.register_workload
    def bigbird_qk_SDDMM_global_A1(N, L, B, H, W, R, G, dtype):
        X = te.placeholder((N, L, B, H), name='X', dtype=dtype)
        Y = te.placeholder(
            (N, L + W // 2 + W // 2, B, H), name='Y', dtype=dtype)
        k = te.reduce_axis((0, H), name='k')
        out_shape = (N, L - 2 * G, B, B)

        def qk_sddmm(n, l, i, j):
            return te.sum(X[n, l + G, i, k] * Y[n, W // 2, j % B, k], axis=k)

        Res = te.compute(
            out_shape,
            qk_sddmm,
            name='Res',
        )

        return [X, Y, Res]

    @auto_scheduler.register_workload
    def bigbird_qk_SDDMM_global_A2(N, L, B, H, W, R, G, dtype):
        X = te.placeholder((N, L, B, H), name='X', dtype=dtype)
        Y = te.placeholder(
            (N, L + W // 2 + W // 2, B, H), name='Y', dtype=dtype)
        k = te.reduce_axis((0, H), name='k')
        out_shape = (N, L - 2 * G, B, B)

        def qk_sddmm(n, l, i, j):
            return te.sum(X[n, l + G, i, k] * Y[n, L - 1, j % B, k], axis=k)

        Res = te.compute(
            out_shape,
            qk_sddmm,
            name='Res',
        )

        return [X, Y, Res]

    args = (batch_size, len // block_size, block_size, hidden_size,
            window_size, random_size, global_size, dtype)

    tasks = [
        tvm.auto_scheduler.SearchTask(
            func=bigbird_qk_SDDMM_global_window, args=args, target=target),
        # tvm.auto_scheduler.SearchTask(
        #     func=bigbird_qk_SDDMM_global , args=args, target=target),
        tvm.auto_scheduler.SearchTask(
            func=bigbird_qk_SDDMM_global_A1, args=args, target=target),
        tvm.auto_scheduler.SearchTask(
            func=bigbird_qk_SDDMM_global_A2, args=args, target=target),
        tvm.auto_scheduler.SearchTask(
            func=bigbird_qk_SDDMM_window, args=args, target=target),
    ]

    shape = (batch_size, len, block_size, hidden_size, window_size,
             global_size)

    log_file = f'ansor.{shape}.json'

    tuner = auto_scheduler.TaskScheduler(tasks)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=64 * 4,
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
        [batch_size, len // block_size, random_size, block_size, hidden_size],
        dtype="float32")
    GatheredV = np.ones(
        [batch_size, len // block_size, random_size, block_size, hidden_size],
        dtype="float32")

    Q_tvm = tvm.nd.array(Q, device=dev)
    K_tvm = tvm.nd.array(K, device=dev)
    V_tvm = tvm.nd.array(V, device=dev)

    global_window_weight = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         (2 + window_size) * block_size),
        device=dev)
    global_weight_A1 = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         block_size),
        device=dev)
    global_weight_A2 = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         block_size),
        device=dev)
    window_weight = tvm.nd.empty(
        (batch_size, len // block_size - 2 * global_size, block_size,
         (window_size) * block_size),
        device=dev)

    inputs_funcs = [(Q_tvm, K_tvm, global_window_weight), (Q_tvm, K_tvm,
                                                           global_weight_A1),
                    (Q_tvm, K_tvm, global_weight_A2), (Q_tvm, K_tvm,
                                                       window_weight)]

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

    file_name = "test_tvm_data"
    with open(file_name, 'a', encoding='utf-8') as f:
        f.writelines(f"{batch_size}_{hidden_size}_{block_size}\n:")
        f.writelines(f'Time breakdown (ms):, {time_log}\n')

    return 0


if __name__ == '__main__':
    len = 4096
    global_size = 1
    window_size = 3
    random_size = 3

    batch_size = 1
    hidden_size = 512
    block_size = 64

    tvm_bigbird(batch_size, hidden_size, len, block_size, window_size,
                random_size, global_size)
