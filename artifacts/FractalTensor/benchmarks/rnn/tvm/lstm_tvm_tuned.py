import os
import sys
import logging
import shutil
from time import time
import logging
import argparse

import torch
from torch import nn

import tvm
import numpy as np
from tvm import relay
from tvm import auto_scheduler
from tvm.contrib import graph_executor
from tvm.relay import frontend

DTYPE = 'float16'

logger = logging.getLogger()
logging.basicConfig(
    level=logging.CRITICAL,
    filename='logs/tvm_eval_log.txt',
    filemode='w',
    format='%(message)s')


def parse_test_args():
    parser = argparse.ArgumentParser(description='Girdlstm')
    parser.add_argument(
        '--seq_len', type=int, help='Sequence length', default=32)
    parser.add_argument(
        '--batch_size', type=int, help='Batch size', default=256)
    parser.add_argument(
        '--hidden_size', type=int, help='Hidden size', default=256)
    parser.add_argument('--depth', type=int, help='Depth size', default=8)
    return parser.parse_args()


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, depth):
        super(StackedLSTM, self).__init__()

        print((f'input_size = {input_size}, '
               f'hidden_size = {hidden_size}, depth = {depth}'))

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.cell = nn.LSTM(
            input_size, hidden_size, num_layers=depth, dtype=torch.float16)

    def forward(self, x):
        out, _ = self.cell(x)
        return out


def test_LSTM(input_size, hidden_size, depth):
    device = 'cuda:0'
    seq_length = 50
    batch_size = 16

    x = torch.randn(seq_length, batch_size, input_size, device=device)
    rnn = StackedLSTM(input_size, hidden_size, depth).to(device)
    y = rnn(x)


def get_model(input_size,
              hidden_size,
              depth,
              seq_length,
              batch_size,
              device=torch.device('cuda:0')):
    torch_model = StackedLSTM(input_size, hidden_size, depth)

    x_shape = [seq_length, batch_size, input_size]
    x = torch.randn(*x_shape, dtype=torch.float16)

    # set the model to inference mode
    torch_model.eval()

    shape_desc = [('input', x_shape)]
    model = torch.jit.trace(torch_model.to(device), x.to(device)).eval()
    mod, params = relay.frontend.from_pytorch(model, shape_desc)

    print((f'd={depth}, bs={batch_size}, seq={seq_length}, '
           f'is={input_size}, hs={hidden_size}: '
           'load from frontend successfully!'))
    return mod, params


def tune_and_export(log_file, code_file, lib_file, mod, params, target,
                    batch_size, seq_length, input_size, hidden_size, depth):
    # extract tasks from network
    start = time()
    tasks, task_weights = auto_scheduler.extract_tasks(mod['main'], params,
                                                       target)
    search = time()
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    print((f'd={depth}, bs={batch_size}, seq={seq_length}, '
           f'is={input_size}, hs={hidden_size}: Schedule tasks successfully!'))
    print('total schedule time: %.6f s\n' % (search - start))
    for idx, task in enumerate(tasks):
        print('========== Task %d  (workload key: %s) ==========' %
              (idx, task.workload_key))
        print(task.compute_dag)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )
    tuner.tune(tune_option)
    tune = time()
    print((f'd={depth},bs={batch_size}, seq={seq_length}, '
           f'is={input_size}, hs={hidden_size}: Tune successfully!'))
    print('total tuning time: %.6f s\n' % (tune - search))
    with auto_scheduler.ApplyHistoryBest(log_file):
        # build library to export
        with tvm.transform.PassContext(
                opt_level=3, config={'relay.backend.use_auto_scheduler':
                                     True}):
            lib = relay.build(mod, target=target, params=params)
            built = time()

        # build module for code
        with relay.build_config(opt_level=3):
            graph, lib_c, params = relay.build_module.build(
                mod, target, params=params)

    with open(code_file, 'w') as f:
        lib.export_library(lib_file)
        print(lib_c.imported_modules[0].get_source(), file=f)
        f.write(('\n total schedule time: %.6f s, tuning time: %.6f s, '
                 'building time: %.6f s\n') % (search - start, tune - search,
                                               built - tune))
    print('Tune and export the library successfully!\n')


def evaluate_kernel(target: str, lib_file: str, batch_size: int,
                    seq_length: int, input_size: int, depth: int):
    dev = tvm.device(str(target), 0)
    lib: tvm.runtime.Module = tvm.runtime.load_module(lib_file)
    module = graph_executor.GraphModule(lib['default'](dev))
    x_tvm = tvm.nd.array(
        (np.random.uniform(size=(seq_length, batch_size,
                                 input_size))).astype(DTYPE))
    module.set_input('input', x_tvm)

    logger.critical(
        f'd={depth},bs={batch_size}, seq={seq_length}, hs={input_size}')
    logger.critical(module.benchmark(dev, number=10, repeat=50))


def tvm_tuned_stacked_lstm(batch_size: int, seq_length: int, hidden_size: int,
                           input_size: int, depth: int, log_dir: str,
                           code_dir: str, lib_dir: str):
    prefix = f'lstm_b{batch_size}_s{seq_length}_h{hidden_size}_d{depth}'
    log_file = os.path.join(log_dir, prefix + '.json')
    code_file = os.path.join(code_dir, prefix + '.cu')
    lib_file = os.path.join(lib_dir, prefix + '.so')

    # Test correctness.
    # test_LSTM(input_size=64, hidden_size=32, depth=2)

    target = tvm.target.Target(target='cuda', host='llvm')
    mod, params = get_model(
        input_size=input_size,
        hidden_size=hidden_size,
        depth=depth,
        seq_length=seq_length,
        batch_size=batch_size)
    open(os.path.join(log_dir, prefix + '_mode.txt'), 'w').write(str(mod))

    if not os.path.exists(lib_file):
        tune_and_export(
            log_file,
            code_file,
            lib_file,
            mod,
            params,
            target,
            batch_size=batch_size,
            seq_length=seq_length,
            input_size=input_size,
            hidden_size=hidden_size,
            depth=depth)

    dev = tvm.device(str(target), 0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
        lib.export_library(lib_file)

    evaluate_kernel(
        target,
        lib_file,
        batch_size=batch_size,
        seq_length=seq_length,
        input_size=input_size,
        depth=depth)


if __name__ == '__main__':
    log_dir = 'logs/search_log'
    code_dir = 'logs/generated_code'
    lib_dir = 'logs/lib'

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if not os.path.isdir(code_dir):
        os.mkdir(code_dir)
    if not os.path.isdir(lib_dir):
        os.mkdir(lib_dir)

    cmd_args = parse_test_args()
    seq_length = cmd_args.seq_len
    batch_size = cmd_args.batch_size
    hidden_size = cmd_args.hidden_size
    depth = cmd_args.depth

    tvm_tuned_stacked_lstm(batch_size, seq_length, hidden_size, hidden_size,
                           depth, log_dir, code_dir, lib_dir)
