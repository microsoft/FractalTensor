import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import unittest
from time import time

import torch
from triton_model import StackedDRNN

from utils import *


class TritonDRNN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.log_dir = ''

    def _apply_forward(self, test_name, test_case, model, *inputs):
        for i in range(WARMUP):
            output = model(*inputs)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()

        for i in range(ITERS):
            output = model(*inputs)

        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / ITERS
        report(test_name, test_case, OUTPUT_FILE, elapsed)

    def test_drnn_forward(self):
        if not DEFAULT_TEST:
            shape = (SEQ_LEN, BATCH_SIZE, INPUT_SIZE)
            for device in [
                    # 'cpu',
                    'cuda:0',
            ]:
                x = torch.randn(*shape, device=device, dtype=torch.float16)
                net = StackedDRNN(
                    batch_size=BATCH_SIZE,
                    seq_len=SEQ_LEN,
                    input_size=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    dilation=DILATION,
                    device=device,
                    dtype=torch.float16).to(device)
                net.eval()

                script_module = net
                test_name = f'Triton_Stacked_DLSTM_{device}'
                test_case = [SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS]
                self._apply_forward(test_name, test_case, script_module, x)

    # def test_drnn_pad_per_layer_forward(self):
    #     shape = (SEQ_LEN, BATCH_SIZE, INPUT_SIZE)

    #     for device in [
    #             # 'cpu',
    #             'cuda:0',
    #     ]:
    #         x = torch.randn(shape, device=device, dtype=torch.float16)

    #         net = StackedDRNN(
    #             batch_size=BATCH_SIZE,
    #             seq_len=SEQ_LEN,
    #             input_size=INPUT_SIZE,
    #             hidden_size=HIDDEN_SIZE,
    #             dilation=DILATION,
    #             device=device).to(device)
    #         net.eval()

    #         rate = DILATION[-1]
    #         padding_data = torch.zeros(
    #             (rate - (SEQ_LEN % rate)) % rate,  # padding number
    #             BATCH_SIZE,
    #             INPUT_SIZE,
    #             device=device,
    #             dtype=torch.float16)

    #         test_name = f'Triton_Stacked_DLSTM_pad_per_layer_{device}'
    #         test_case = [SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, NUM_LAYERS]
    #         self._apply_forward(test_name, test_case, net, x)

    def test_default_data(self):
        if DEFAULT_TEST:
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
                    net = StackedDRNN(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        input_size=hidden,
                        hidden_size=hidden,
                        dilation=DILATION[0:num_layers],
                        device=device,
                        dtype=torch.float16).to(device)
                    net.eval()

                    script_module = net
                    return x, script_module

                test_cases = [
                    # overall
                    [50, 256, 256, 6],
                    [50, 256, 512, 6],
                    [50, 256, 1024, 6],
                    # scale with depth
                    [50, 256, 256, 1],
                    [50, 256, 256, 2],
                    [50, 256, 256, 3],
                    [50, 256, 256, 4],
                    [50, 256, 256, 5],
                    [50, 256, 256, 6],
                    [50, 256, 1024, 1],
                    [50, 256, 1024, 2],
                    [50, 256, 1024, 3],
                    [50, 256, 1024, 4],
                    [50, 256, 1024, 5],
                    [50, 256, 1024, 6],
                    # scale with seq
                    [32, 256, 256, 6],
                    [64, 256, 256, 6],
                    [128, 256, 256, 6],
                    [32, 256, 1024, 6],
                    [64, 256, 1024, 6],
                    [128, 256, 1024, 6],
                ]

                for test_case in test_cases:
                    x, script_module = build_data(test_case)
                    self._apply_forward(test_name, test_case, script_module, x)
                    del x
                    del script_module
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    if OUTPUT_FILE:
        with open(OUTPUT_FILE, 'w') as fout:
            fout.write(
                "depth\t[seq_length, batch_size, hidden_size]\tTriton(ms)\n")
    unittest.main(argv=['first-arg-is-ignored'])
