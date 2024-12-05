from time import time
import unittest
import torch
import logging
import argparse
from pt_model import StackedGridModel

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


class PytorchGrid(unittest.TestCase):
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
                "depth\t[seq_length, batch_size, hidden_size]\tPyTorch(ms)\n")

    LOG_DEBUG_INFO = 1
    PROFILER_ENABLE = 0

    def setUp(self):
        self.shape = (PytorchGrid.SEQ_LEN, PytorchGrid.BATCH_SIZE,
                      PytorchGrid.HIDDEN_SIZE)

    def _report(self, test_name, test_case, elapsed):
        seq_len, batch_size, hidden, num_layers = test_case

        print(f"\nbench-grid\tdepth\t{num_layers}\tseq_length\t{seq_len}\t"
              f"batch_size\t{batch_size}\t"
              f"hidden_size\t{hidden}\tPyTroch(ms)\t{elapsed}")

        if self.OUTPUT_FILE:
            with open(self.OUTPUT_FILE, 'a') as fout:
                fout.write(
                    f"{num_layers}\t[{seq_len}, {seq_len}, {batch_size}, {hidden}]\t"
                    f"{elapsed}\n")

    def _apply_forward(self, test_name, test_case, source, target, model):

        for i in range(PytorchGrid.WARM_UP):
            output = model(source, target)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        if PytorchGrid.PROFILER_ENABLE:
            with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True) as prof:
                with record_function("model_inference"):
                    for i in range(PytorchGrid.ITERS):
                        output = model(source, target)

            prof.export_chrome_trace("trace_" + test_name + ".json")
        else:
            for i in range(PytorchGrid.ITERS):
                output = model(source, target)

        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / PytorchGrid.ITERS

        self._report(test_name, test_case, elapsed)

    def test_grid_forward(self):
        if not self.DEFAULT_TEST:
            for enable_jit in [
                    # False,
                    True,
            ]:
                for device in [
                        "cuda:0",
                        # "cpu",
                ]:
                    target = torch.randn(*self.shape, device=device)
                    source = torch.randn(*self.shape, device=device)
                    model = StackedGridModel(
                        PytorchGrid.DEPTH,
                        PytorchGrid.SEQ_LEN,
                        PytorchGrid.SEQ_LEN,
                        PytorchGrid.BATCH_SIZE,
                        PytorchGrid.HIDDEN_SIZE,
                        device,
                        enable_jit=enable_jit).to(device)
                    test_name = f"gridlstm_{device}_forward" + ("_JIT"
                                                                if enable_jit
                                                                else "")
                    test_case = [
                        PytorchGrid.SEQ_LEN, PytorchGrid.BATCH_SIZE,
                        PytorchGrid.HIDDEN_SIZE, PytorchGrid.DEPTH
                    ]
                    self._apply_forward(test_name, test_case, source, target,
                                        model)

    def test_default_data(self):
        if self.DEFAULT_TEST:
            for device in [
                    "cuda:0",
                    # "cpu",
            ]:
                test_name = f"gridlstm_{device}_forward_JIT"
                print("default test:", test_name)

                def build_data(test_case):
                    seq_len, batch_size, hidden, num_layers = test_case
                    target = torch.randn(
                        (seq_len, batch_size, hidden),
                        device=device,
                    )
                    source = torch.randn(
                        (seq_len, batch_size, hidden),
                        device=device,
                    )
                    model = StackedGridModel(
                        num_layers,
                        seq_len,
                        seq_len,
                        batch_size,
                        hidden,
                        device,
                        enable_jit=True).to(device)
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
