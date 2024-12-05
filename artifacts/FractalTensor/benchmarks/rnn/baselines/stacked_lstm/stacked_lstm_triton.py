from collections import namedtuple
import torch
from time import time
import argparse
import unittest
import logging
import sys
import os

import triton_model as model
from torch.profiler import profile
from torch.profiler import record_function
from torch.profiler import ProfilerActivity

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
    parser = argparse.ArgumentParser(description='Girdlstm')
    parser.add_argument(
        '--seq_len', type=int, help='Sequence length', default=32)
    parser.add_argument(
        '--batch_size', type=int, help='Batch size', default=256)
    parser.add_argument(
        '--hidden_size', type=int, help='Hidden size', default=256)
    parser.add_argument('--depth', type=int, help='Depth size', default=8)
    parser.add_argument(
        '--output_file', type=str, help='Output file path', default=None)
    parser.add_argument(
        '--default_test',
        type=str2bool,
        help='Whether to run the default test',
        default=False)
    return parser.parse_args()


class TritonStackedLSTM(unittest.TestCase):
    WARM_UP = 5
    ITERS = 10
    PROFILER_ENABLE = False
    dtype = torch.float16

    cmd_args = parse_test_args()
    SEQ_LEN = cmd_args.seq_len
    BATCH_SIZE = cmd_args.batch_size
    HIDDEN = cmd_args.hidden_size
    NUM_LAYERS = cmd_args.depth
    OUTPUT_FILE = cmd_args.output_file
    DEFAULT_TEST = cmd_args.default_test

    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'w') as fout:
            fout.write(
                "depth\t[seq_length, batch_size, hidden_size]\tTriton(ms)\n")

    def setUp(self):
        torch.manual_seed(1234)

    # def _report(self, test_name, test_case, start):
    #     seq_len, batch_size, hidden, num_layers = test_case
    #     torch.cuda.synchronize()
    #     elapsed_time = time() - start
    #     average_time = elapsed_time / TritonStackedLSTM.ITERS
    #     seq_per_sec = (TritonStackedLSTM.ITERS *
    #                    TritonStackedLSTM.BATCH_SIZE) / elapsed_time

    #     print(
    #         f"depth: {num_layers}, seq_length: {seq_len}, batch_size: {batch_size}, "
    #         f"hidden_size: {hidden}, Triton(ms): {average_time * 1000}ms")

    #     if self.OUTPUT_FILE:
    #         with open(self.OUTPUT_FILE, 'a') as fout:
    #             fout.write(
    #                 f"{num_layers}\t[{seq_len}, {batch_size}, {hidden}]\t"
    #                 f"{average_time * 1000}\n")

    def _report(self, test_name, test_case, elapsed):
        seq_len, batch_size, hidden, num_layers = test_case
        torch.cuda.synchronize()
        # elapsed_time = time() - start
        # average_time = elapsed_time / TritonStackedLSTM.ITERS
        # seq_per_sec = (TritonStackedLSTM.ITERS *
        #                TritonStackedLSTM.BATCH_SIZE) / elapsed_time

        print(
            f"depth: {num_layers}, seq_length: {seq_len}, batch_size: {batch_size}, "
            f"hidden_size: {hidden}, Triton(ms): {elapsed}ms")

        if self.OUTPUT_FILE:
            with open(self.OUTPUT_FILE, 'a') as fout:
                fout.write(
                    f"{num_layers}\t[{seq_len}, {batch_size}, {hidden}]\t"
                    f"{elapsed}\n")

    def _apply_forward(self, test_name, test_case, x, model):
        model.eval()
        for i in range(TritonStackedLSTM.WARM_UP):
            output = model(x)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        if TritonStackedLSTM.PROFILER_ENABLE:
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True,
                    record_shapes=True) as prof:
                with record_function('model_inference'):
                    for i in range(TritonStackedLSTM.ITERS):
                        if i >= 5:
                            break
                        output = model(x)

            print(prof.key_averages().table(
                sort_by='cuda_time_total', row_limit=20))
            print(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by='cuda_time_total', row_limit=20))
        else:
            for i in range(TritonStackedLSTM.ITERS):
                output = model(x)

            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(
                end_event) / TritonStackedLSTM.ITERS
            self._report(test_name, test_case, elapsed)

    def test_fine_grained_op_lstm_forward(self):
        if not self.DEFAULT_TEST:
            for device in [
                    # 'cpu',
                    'cuda:0',
            ]:
                x = torch.randn(
                    (TritonStackedLSTM.SEQ_LEN, TritonStackedLSTM.BATCH_SIZE,
                     TritonStackedLSTM.HIDDEN),
                    device=device,
                    dtype=TritonStackedLSTM.dtype)

                m = model.StackedLSTM(
                    batch_size=TritonStackedLSTM.BATCH_SIZE,
                    max_seq_length=TritonStackedLSTM.SEQ_LEN,
                    hidden_size=TritonStackedLSTM.HIDDEN,
                    num_layers=TritonStackedLSTM.NUM_LAYERS,
                    device=device,
                    dtype=TritonStackedLSTM.dtype)

                test_name = f'triton_finegrained_op_{device}'
                test_case = [
                    TritonStackedLSTM.SEQ_LEN, TritonStackedLSTM.BATCH_SIZE,
                    TritonStackedLSTM.HIDDEN, TritonStackedLSTM.NUM_LAYERS
                ]
                print("Triton for Stacked LSTM:", test_name)
                self._apply_forward(test_name, test_case, x, m)

    def test_default_data(self):
        if self.DEFAULT_TEST:
            for device in [
                    # 'cpu',
                    'cuda:0',
            ]:
                test_name = f'triton_finegrained_op_{device}'
                print("default test:", test_name)

                def build_data(test_case):
                    seq_len, batch_size, hidden, num_layers = test_case
                    x = torch.randn(
                        (seq_len, batch_size, hidden),
                        device=device,
                        dtype=torch.float16)
                    m = model.StackedLSTM(
                        batch_size=batch_size,
                        max_seq_length=seq_len,
                        hidden_size=hidden,
                        num_layers=num_layers,
                        device=device,
                        dtype=torch.float16)
                    return x, m

                test_cases = [
                    # overall
                    # [seq_len, batch_size, hidden, num_layers]
                    [64, 256, 256, 32],
                    [64, 256, 512, 32],
                    [64, 256, 1024, 32],
                    # scale with depth
                    [64, 256, 256, 1],
                    [64, 256, 256, 2],
                    [64, 256, 256, 4],
                    [64, 256, 256, 8],
                    [64, 256, 256, 16],
                    [64, 256, 256, 32],
                    [64, 256, 1024, 1],
                    [64, 256, 1024, 2],
                    [64, 256, 1024, 4],
                    [64, 256, 1024, 8],
                    [64, 256, 1024, 16],
                    [64, 256, 1024, 32],
                    # scale with length
                    [32, 256, 256, 32],
                    [64, 256, 256, 32],
                    [128, 256, 256, 32],
                    [32, 256, 1024, 32],
                    [64, 256, 1024, 32],
                    [128, 256, 1024, 32],
                    # figure 2
                    [64, 256, 256, 1],
                    [64, 256, 256, 4],
                    [64, 256, 256, 8],
                    [64, 256, 256, 12],
                    [64, 256, 256, 16],
                    [64, 256, 256, 20],
                ]

                for test_case in test_cases:
                    x, m = build_data(test_case)
                    self._apply_forward(test_name, test_case, x, m)
                    del x
                    del m
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'])
