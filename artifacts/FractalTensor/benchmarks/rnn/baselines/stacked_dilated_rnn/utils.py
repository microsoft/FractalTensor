import os
import logging
import argparse
from time import time


def max_dilation(depth):
    depth = int(depth)
    if depth > 6:
        raise argparse.ArgumentTypeError(
            'To avoid memory errors, %s should not be too large.' % depth)
    return depth


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
        '--seq_len', type=int, help='Sequence length', default=50)
    parser.add_argument(
        '--batch_size', type=int, help='Batch size', default=256)
    parser.add_argument(
        '--hidden_size', type=int, help='Hidden size', default=256)
    parser.add_argument(
        '--input_size', type=int, help='Input size', default=256)
    parser.add_argument(
        '--depth', type=max_dilation, help='Depth size', default=1)
    parser.add_argument(
        '--output_file', type=str, help='Output file path', default=None)
    parser.add_argument(
        '--default_test',
        type=str2bool,
        help='Whether to run the default test',
        default=False)
    return parser.parse_args()


cmd_args = parse_test_args()
SEQ_LEN = cmd_args.seq_len
BATCH_SIZE = cmd_args.batch_size
HIDDEN_SIZE = cmd_args.hidden_size
INPUT_SIZE = cmd_args.input_size
NUM_LAYERS = cmd_args.depth
OUTPUT_FILE = cmd_args.output_file
DEFAULT_TEST = cmd_args.default_test

DILATION = [1, 2, 4, 8, 16, 32]
if not DEFAULT_TEST:
    DILATION = DILATION[0:NUM_LAYERS]

ITERS = 10
WARMUP = 5
LOG_DEBUG_INFO = 0

print_header = True


def report(test_name, test_case, OUTPUT_FILE, elapsed):
    seq_len, batch_size, hidden, num_layers = test_case
    # elapsed_time = time() - start
    # average_time = elapsed_time / ITERS
    # seq_per_sec = (ITERS * BATCH_SIZE) / elapsed_time

    print(
        f"depth: {num_layers}, seq_length: {seq_len}, batch_size: {batch_size}, "
        f"hidden_size: {hidden}, Baseline(ms): {elapsed}ms")

    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'a') as fout:
            fout.write(f"{num_layers}\t[{seq_len}, {batch_size}, {hidden}]\t"
                       f"{elapsed}\n")
