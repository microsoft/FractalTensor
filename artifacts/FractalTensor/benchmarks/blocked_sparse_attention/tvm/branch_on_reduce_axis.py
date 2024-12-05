import timeit
import time
import numpy as np

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
sys.path.append('../..')

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

target_name = 'cuda -libs=cublas'
target = tvm.target.Target(target_name)

batch_size = 8
heads = 1
seq_len = 4096
hidden_size = 512
block_size = 32

target = tvm.target.Target(target_name)
dtype, itype = 'float32', 'int32'


@auto_scheduler.register_workload
def test(B, M, N, K, dtype, itype):
    X = te.placeholder((B, M, K), name='X', dtype=dtype)
    Y = te.placeholder((B, N, K), name='Y', dtype=dtype)
    Rand = te.placeholder((B, N), name='Rand', dtype=itype)
    k = te.reduce_axis((0, K), name='k')
    out_shape = (B, M, N)

    def algorithm(x, i, j):
        return te.sum(X[x, i, k] * Y[x, Rand[x, j], k], axis=k)

    R = te.compute(
        out_shape,
        algorithm,
        name='R',
    )
    return [X, Y, Rand, R]


################################################################################
tasks = [
    tvm.auto_scheduler.SearchTask(
        func=test, args=(16, 1024, 1024, 1024, dtype, itype), target=target),
]

shape = (batch_size, heads, seq_len, block_size)

log_file = f'ansor.{shape}.json'

tuner = auto_scheduler.TaskScheduler(tasks)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=20,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

tuner.tune(tune_option)

funcs = []
for task in tasks:
    sch, args = task.apply_best(log_file)
    funcs.append(tvm.build(sch, args, target))

dev = tvm.cuda()

Q = np.ones([1024, 1024], dtype="float32")
K = np.ones(
    [1024, 1024],  #padding for window access
    dtype="float32")
Rand = np.ones([1024, 1024], dtype="int32")

Q_tvm = tvm.nd.array(Q, device=dev)
K_tvm = tvm.nd.array(K, device=dev)
Rand_tvm = tvm.nd.array(Rand, device=dev)
res = tvm.nd.empty((1024, 1024), device=dev)

inputs_funcs = [(Q_tvm, K_tvm, Rand_tvm, res)]

for func, inputs in zip(funcs, inputs_funcs):
    func(*inputs)

# Evaluation
warmup_num = 5
test_num = 10
time_log = []
for func, inputs in zip(funcs, inputs_funcs):
    evaluator = func.time_evaluator(func.entry_name, dev, number=warmup_num)
    evaluator(*inputs)
    evaluator = func.time_evaluator(func.entry_name, dev, number=test_num)
    time_ms = np.median(evaluator(*inputs).results) * 1000
    time_log.append(time_ms)
print(f"{warmup_num} warmup, {test_num} repeats for evalution")
print('Time breakdown (ms):', time_log)
print("Average e2e time: %.3f ms" % (sum(time_log)))

block_num = batch_size * heads * (seq_len // block_size - 2) * 4
operations = block_num * (
    4 * block_size * block_size * hidden_size + 2 * block_size * block_size)
operations = operations >> 25
print(f"GFLOPs:{operations/(sum(time_log)/1000*32)}")
