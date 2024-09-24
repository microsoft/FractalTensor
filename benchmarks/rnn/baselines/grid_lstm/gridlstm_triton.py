from time import time
import unittest
import torch
import logging
import argparse

from triton_model import StackedGridModel

from torch.profiler import profile
from torch.profiler import record_function
from torch.profiler import ProfilerActivity


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
        '--seq_len', type=int, help='Sequence length', default=10)
    parser.add_argument(
        '--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument(
        '--hidden_size', type=int, help='Hidden size', default=256)
    parser.add_argument('--depth', type=int, help='Depth size', default=4)
    parser.add_argument(
        '--output_file', type=str, help='Output file path', default=None)
    parser.add_argument(
        '--default_test',
        type=str2bool,
        help='Whether to run the default test',
        default=False)
    return parser.parse_args()


class TritonGrid(unittest.TestCase):
    WARM_UP = 5
    ITERS = 10

    cmd_args = parse_test_args()
    SEQ_LEN = cmd_args.seq_len
    BATCH_SIZE = cmd_args.batch_size
    HIDDEN_SIZE = cmd_args.hidden_size
    DEPTH = cmd_args.depth
    OUTPUT_FILE = cmd_args.output_file
    DEFAULT_TEST = cmd_args.default_test

    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'w') as fout:
            fout.write(
                "depth\t[seq_length, batch_size, hidden_size]\tTriton(ms)\n")

    LOG_DEBUG_INFO = 1
    PROFILER_ENABLE = 0

    def setUp(self):
        self.shape = (TritonGrid.SEQ_LEN, TritonGrid.BATCH_SIZE,
                      TritonGrid.HIDDEN_SIZE)

    #     self._init_logger()

    # def _init_logger(self):
    #     self.logger = logging.getLogger()
    #     logging.basicConfig(
    #         level=(logging.DEBUG
    #                if TritonGrid.LOG_DEBUG_INFO else logging.INFO),
    #         filename="grid_lstm_results_triton.txt",
    #         filemode="w",
    #         format="%(message)s")

    def _report(self, test_name, test_case, elapsed):
        seq_len, batch_size, hidden, num_layers = test_case
        # elapsed_time = time() - start
        # average_time = elapsed_time / TritonGrid.ITERS
        # seq_per_sec = (TritonGrid.ITERS * TritonGrid.BATCH_SIZE) / elapsed_time

        print(
            f"depth: {num_layers}, seq_length: {seq_len}, batch_size: {batch_size}, "
            f"hidden_size: {hidden}, Triton(ms): {elapsed}ms")

        if self.OUTPUT_FILE:
            with open(self.OUTPUT_FILE, 'a') as fout:
                fout.write(
                    f"{num_layers}\t[{seq_len}, {seq_len}, {batch_size}, {hidden}]\t"
                    f"{elapsed}\n")

    def _apply_forward(self, test_name, test_case, source, target, model):

        for i in range(TritonGrid.WARM_UP):
            output = model(source, target)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        if TritonGrid.PROFILER_ENABLE:
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True) as prof:
                with record_function("model_inference"):
                    for i in range(TritonGrid.ITERS):
                        output = model(source, target)

            prof.export_chrome_trace("trace_" + test_name + ".json")
        else:
            for i in range(TritonGrid.ITERS):
                output = model(source, target)

        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / TritonGrid.ITERS
        self._report(test_name, test_case, elapsed)

    def test_grid_forward(self):
        if not self.DEFAULT_TEST:
            for device in [
                    "cuda:0",
                    # "cpu",
            ]:
                target = torch.randn(*self.shape, device=device)
                source = torch.randn(*self.shape, device=device)
                model = StackedGridModel(
                    TritonGrid.DEPTH, TritonGrid.SEQ_LEN, TritonGrid.SEQ_LEN,
                    TritonGrid.BATCH_SIZE, TritonGrid.HIDDEN_SIZE,
                    device).to(device)
                test_name = f"gridlstm_{device}_forward"
                test_case = [
                    TritonGrid.SEQ_LEN, TritonGrid.BATCH_SIZE,
                    TritonGrid.HIDDEN_SIZE, TritonGrid.DEPTH
                ]
                self._apply_forward(test_name, test_case, source, target,
                                    model)

    def test_default_data(self):
        if self.DEFAULT_TEST:
            for device in [
                    "cuda:0",
                    # "cpu",
            ]:
                test_name = f"gridlstm_{device}_forward"
                print("default test:", test_name)

                def build_data(test_case):
                    seq_len, batch_size, hidden, num_layers = test_case
                    target = torch.randn(
                        (seq_len, batch_size, hidden),
                        device=device,
                        # dtype=torch.float16
                    )
                    source = torch.randn(
                        (seq_len, batch_size, hidden),
                        device=device,
                        # dtype=torch.float16
                    )
                    model = StackedGridModel(num_layers, seq_len, seq_len,
                                             batch_size, hidden,
                                             device).to(device)
                    return target, source, model

                test_cases = [
                    # overall
                    # [seq_len, batch_size, hidden, num_layers]
                    [10, 32, 256, 32],
                    [10, 32, 512, 32],
                    [10, 32, 1024, 32],
                    # scale with depth
                    [10, 32, 256, 1],
                    [10, 32, 256, 2],
                    [10, 32, 256, 4],
                    [10, 32, 256, 8],
                    [10, 32, 256, 16],
                    [10, 32, 256, 32],
                    [10, 32, 1024, 1],
                    [10, 32, 1024, 2],
                    [10, 32, 1024, 4],
                    [10, 32, 1024, 8],
                    [10, 32, 1024, 16],
                    [10, 32, 1024, 32],
                    # scale with length
                    [5, 32, 256, 32],
                    [7, 32, 256, 32],
                    [10, 32, 256, 32],
                    [5, 32, 1024, 32],
                    [7, 32, 1024, 32],
                    [10, 32, 1024, 32],
                ]

                for test_case in test_cases:
                    target, source, model = build_data(test_case)
                    self._apply_forward(test_name, test_case, source, target,
                                        model)
                    del target
                    del source
                    del model
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'])
