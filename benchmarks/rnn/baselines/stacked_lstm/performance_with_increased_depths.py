# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import math
import os
import sys
from collections import namedtuple
from time import time

import pt_model as model
import tensorflow as tf
import test_utils as tu
import torch
from tf_model.rnn import FineGrainedOpLstmNet, StaticRNN, WhileOpLstmNet

torch.manual_seed(1234)

# supress tensorflow deprecation warning.
tf.get_logger().setLevel('ERROR')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ITERS = 30
BATCH_SIZE = 32
HIDDEN = 128
SEQ_LEN = 30

logger = logging.getLogger()

logging.basicConfig(level=logging.INFO,
                    filename='figures/perf_with_increased_depth.tsv',
                    filemode='w',
                    format='%(message)s')
logger.info(('Depth\tTestName\tAvgTime\tThroughput\tRatio'))


def GetTestName(cell_type):
    if cell_type == 'cudnn_lstm':
        return 'CuDNN'
    elif cell_type == 'v2':
        return 'PT_JITed'


def RunPyTorchTest(batch_size, seq_len, hidden, depth, cell_type):
    input_shape = [seq_len, batch_size, hidden]
    torch.backends.cudnn.enabled = True
    device = 'cuda:0'

    x = torch.randn(*input_shape, device=device)

    m = model.small_model(batch_size=batch_size,
                          cell_type=cell_type,
                          max_seq_length=seq_len,
                          hidden_size=hidden,
                          num_layers=depth).to(device)
    m = torch.jit.script(m)
    m.eval()

    torch.cuda.synchronize()
    for i in range(10):  # warmup
        output = m(x)

    torch.cuda.synchronize()
    start = time()
    for i in range(ITERS):
        output = m(x)
    return time() - start  # count in seconds


def RunTensorFlowGraphTest(model, batch_size, seq_len, hidden, depth):
    dev = 'gpu'
    stddev = 1.0 / math.sqrt(hidden)

    with tf.device(tu.device(dev)):
        data = tf.random.uniform((seq_len, batch_size, hidden),
                                 minval=-stddev,
                                 maxval=stddev)

        output = model(data)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            for _ in range(5):  # warmup
                sess.run(output)

            start = time()
            for _ in range(ITERS):
                sess.run(output)
    return time() - start


def RunTensorFlowEagerAutoGraphTest(model, batch_size, seq_len, hidden, depth):
    dev = 'gpu'
    stddev = 1.0 / math.sqrt(hidden)

    with tf.device(tu.device(dev)):
        data = tf.random.uniform((seq_len, batch_size, hidden),
                                 minval=-stddev,
                                 maxval=stddev)

        for i in range(5):  # warmup
            y = model(data)
        gc.collect()

        start = time()
        for i in range(ITERS):
            y = model(data)
    return time() - start


def report(test_name, total_times):
    throughputs = [BATCH_SIZE * ITERS / t for t in total_times]

    base = total_times[0]
    raitos = []
    for t in total_times:
        raitos.append(t / base)

    for i, (time, throughput,
            ratio) in enumerate(zip(total_times, throughputs, raitos)):
        logger.info('%d\t%s\t%.5f\t%.5f\t%.5f' %
                    (i + 1, test_name, time / ITERS * 1000, throughput, ratio))


if __name__ == '__main__':
    max_depth = 20

    for cell_type in [
            'v2',
            'cudnn_lstm',
    ]:
        total_times = []
        for depth in range(1, max_depth + 1):
            print(f'{GetTestName(cell_type)}, depth = {depth}')
            t = RunPyTorchTest(BATCH_SIZE, SEQ_LEN, HIDDEN, depth, cell_type)
            total_times.append(t)

        report(GetTestName(cell_type), total_times)

    tf.compat.v1.disable_eager_execution()

    total_times = []
    for depth in range(1, max_depth + 1):
        print(f'depth = {depth}')
        model = WhileOpLstmNet(HIDDEN, HIDDEN, depth)
        t = RunTensorFlowGraphTest(model, BATCH_SIZE, SEQ_LEN, HIDDEN, depth)
        total_times.append(t)
    report('TF_WhileOpLSTM', total_times)

    total_times = []
    for depth in range(1, max_depth + 1):
        print(f'depth = {depth}')
        model = FineGrainedOpLstmNet(HIDDEN, HIDDEN, depth, 'v2')
        t = RunTensorFlowGraphTest(model, BATCH_SIZE, SEQ_LEN, HIDDEN, depth)
        total_times.append(t)
    report('TF_GraphMode', total_times)

    total_times = []
    for depth in range(1, max_depth + 1):
        model = StaticRNN(hidden_size=HIDDEN,
                          num_layers=depth,
                          use_cudnn_rnn=False)
        t = RunTensorFlowGraphTest(model, BATCH_SIZE, SEQ_LEN, HIDDEN, depth)
        total_times.append(t)
    report('TF_StaticLSTMCell', total_times)

    tf.compat.v1.enable_eager_execution()
    total_times = []

    # for depth in range(1, max_depth + 1):
    #     print(f'depth = {depth}')
    #     # model = FineGrainedOpLstmNet(HIDDEN, HIDDEN, depth, 'v2')
    #     model = StaticRNN(
    #         hidden_size=HIDDEN, num_layers=depth, use_cudnn_rnn=False)
    #     t = RunTensorFlowEagerAutoGraphTest(model, BATCH_SIZE, SEQ_LEN, HIDDEN,
    #                                         depth)
    #     total_times.append(t)
    #     report('TF_AutoGraph', total_times)
