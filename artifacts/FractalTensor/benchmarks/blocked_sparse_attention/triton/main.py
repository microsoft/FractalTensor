import torch

import triton
import triton.language as tl
import argparse
import types

from time import time
from op import *


class TritonBigbird():
    global_size = 1
    window_size = 3
    random_size = 3
    warm = 5

    device = 'cuda'
    dtype = torch.float32

    def __init__(self, cmd_args):
        self.seq_len = cmd_args.seq_len
        self.batch_size = cmd_args.batch_size
        self.hidden_size = cmd_args.hidden_size
        self.block_size = cmd_args.block_size
        self.block_num = self.seq_len // self.block_size

    def init_random_index(self, random_index):
        for i in range(self.block_num):
            for j in range(self.random_size):
                random_index[i, j] = (
                    (i * self.random_size + j) // self.random_size)

    def test(self):
        padding_size = (self.window_size // 2) * self.block_size
        Q = torch.randn(
            (self.batch_size, self.seq_len, self.hidden_size),
            device=self.device,
            dtype=self.dtype)
        K = torch.randn(
            (self.batch_size, self.hidden_size,
             padding_size + self.seq_len + padding_size),
            device=self.device,
            dtype=self.dtype)
        V = torch.randn(
            (self.batch_size, padding_size + self.seq_len + padding_size,
             self.hidden_size),
            device=self.device,
            dtype=self.dtype)
        QK = torch.zeros(
            [self.batch_size, self.seq_len, self.seq_len],
            device=self.device,
            dtype=self.dtype)
        softmax_QK = torch.zeros(
            [self.batch_size, self.seq_len, self.seq_len],
            device=self.device,
            dtype=self.dtype)
        O = torch.zeros(
            [self.batch_size, self.seq_len, self.hidden_size],
            device=self.device,
            dtype=self.dtype)
        random_index = torch.zeros(
            [self.block_num, self.random_size],
            device=self.device,
            dtype=torch.int32)

        para = (self.batch_size, self.global_size, self.random_size,
                self.window_size, self.hidden_size, self.seq_len,
                self.seq_len // self.block_size, self.block_size)

        for i in range(self.warm):
            global_qk(Q, K, QK, para)
            global_softmax(QK, softmax_QK, para)
            global_wv(softmax_QK, V, O, para)
            sparse_qk(Q, K, QK, random_index, para)
            sparse_softmax(QK, softmax_QK, para)
            sparse_wv(softmax_QK, V, O, random_index, para)

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        Q = torch.randn(
            (self.batch_size, self.seq_len, self.hidden_size),
            device=self.device,
            dtype=self.dtype)
        K = torch.randn(
            (self.batch_size, self.hidden_size,
             padding_size + self.seq_len + padding_size),
            device=self.device,
            dtype=self.dtype)
        V = torch.randn(
            (self.batch_size, padding_size + self.seq_len + padding_size,
             self.hidden_size),
            device=self.device,
            dtype=self.dtype)
        QK = torch.zeros(
            [self.batch_size, self.seq_len, self.seq_len],
            device=self.device,
            dtype=self.dtype)
        softmax_QK = torch.zeros(
            [self.batch_size, self.seq_len, self.seq_len],
            device=self.device,
            dtype=self.dtype)
        O = torch.zeros(
            [self.batch_size, self.seq_len, self.hidden_size],
            device=self.device,
            dtype=self.dtype)
        random_index = torch.zeros(
            [self.block_num, self.random_size],
            device=self.device,
            dtype=torch.int32)
        self.init_random_index(random_index)

        global_qk(Q, K, QK, para)
        global_softmax(QK, softmax_QK, para)
        global_wv(softmax_QK, V, O, para)
        sparse_qk(Q, K, QK, random_index, para)
        sparse_softmax(QK, softmax_QK, para)
        sparse_wv(softmax_QK, V, O, random_index, para)

        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event)

        print(
            f"block_size\t{self.block_size}\tseq_length\t{self.seq_len}, batch_size\t{self.batch_size}\t"
            f"hidden_size\t{self.hidden_size}\tTriton(ms)\t{elapsed}")

        return elapsed


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


if __name__ == "__main__":
    cmd_args = parse_test_args()
    DEFAULT_TEST = cmd_args.default_test
    OUTPUT_FILE = cmd_args.output_file
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'w') as fout:
            fout.write(
                "batch size\t sequence length\thidden\tblock size\telapsed time(ms)\n"
            )
    if not DEFAULT_TEST:
        run_time = TritonBigbird(cmd_args).test()
        output_file(OUTPUT_FILE, cmd_args, run_time)
    else:
        test1_cmd_args = types.SimpleNamespace(
            seq_len=4096, batch_size=32, hidden_size=512, block_size=64)
        run_time = TritonBigbird(test1_cmd_args).test()
        output_file(OUTPUT_FILE, test1_cmd_args, run_time)

        test2_cmd_args = types.SimpleNamespace(
            seq_len=8192, batch_size=32, hidden_size=512, block_size=64)
        run_time = TritonBigbird(test2_cmd_args).test()
        output_file(OUTPUT_FILE, test2_cmd_args, run_time)
