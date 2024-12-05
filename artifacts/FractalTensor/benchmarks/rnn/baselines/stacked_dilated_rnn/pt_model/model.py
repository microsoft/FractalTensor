import math
import random
import numpy as np

from typing import List
from typing import Tuple

import torch
import torch.jit as jit
from torch import Tensor
import torch.nn as nn

__all__ = [
    'StackedDRNNJIT',
    'StackedDRNN',
]


class StackedDRNNJIT(nn.Module):
    def __init__(self,
                 batch_size: int,
                 seq_len: int,
                 input_size: int,
                 hidden_size: int,
                 dilation: List[int],
                 device: str,
                 dtype=torch.float16):
        super(StackedDRNNJIT, self).__init__()

        self.batch_size = batch_size
        self.seq_len = input_size

        rate = dilation[-1]
        self.register_buffer(
            'padding_data',
            torch.zeros(
                (rate - (seq_len % rate)) % rate,
                batch_size,
                input_size,
                device=device,
                dtype=dtype))

        self.dilation_above_first_layer = dilation[1:]
        self.cell1 = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=False,
            dropout=0.,
            dtype=dtype)
        self.cells = torch.nn.ModuleList([
            nn.LSTM(
                input_size,
                hidden_size,
                num_layers=1,
                batch_first=False,
                dropout=0.,
                dtype=dtype) for i in range(len(dilation) - 1)
        ])

    def forward(self, input: Tensor) -> Tensor:
        # step 0: pad the input
        input_x = torch.cat((input, self.padding_data))

        # no special treatment for the first layer.
        xs, _ = self.cell1(input_x)

        for i, cell in enumerate(self.cells):
            # for layers above the frist layer.
            # step 1: pre-process: form a new batch
            xs_splits = xs.split(self.dilation_above_first_layer[i])
            xs_ = torch.jit.annotate(List[Tensor], [])
            for x in xs_splits:
                xs_.append(x.flatten(start_dim=0, end_dim=1))
            dilated_input = torch.stack(xs_)

            # step 2: call LSTM layer
            xs, _ = cell(dilated_input)

            # step 3: post-processing, revert to the original layout
            xss = torch.jit.annotate(List[List[Tensor]], [])
            for x in xs.unbind(0):
                xss.append(torch.split(x, self.batch_size))

            xs_ = torch.jit.annotate(List[Tensor], [])
            for sublist in xss:
                for x in sublist:
                    xs_.append(x)
            xs = torch.stack(xs_)
        return xs


class StackedDRNN(nn.Module):
    def __init__(self,
                 batch_size: int,
                 seq_len: int,
                 input_size: int,
                 hidden_size: int,
                 dilation: List[int],
                 device: str,
                 dtype=torch.float16):
        super(StackedDRNN, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.dtype = dtype

        self.dilation = dilation

        layers = []
        for i in range(len(dilation)):
            c = nn.LSTM(
                self.input_size, self.hidden_size, dropout=0., dtype=dtype)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def _forward(self, input_x, cell, rate):
        L, N, input_size = input_x.size()

        # padding
        pad_num = (rate - (self.seq_len % rate)) % rate
        padding_data = torch.zeros(
            pad_num,
            self.batch_size,
            input_size,
            device=self.device,
            dtype=self.dtype)

        input_x = torch.cat((input_x, padding_data))

        dilated_input = torch.stack(
            tuple(
                map(lambda m: m.flatten(start_dim=0, end_dim=1),
                    input_x.split(rate))),
            dim=0)

        output, _ = cell(dilated_input)

        output_split = [
            torch.split(item, self.batch_size) for item in torch.unbind(output)
        ]

        output_flatten = torch.stack(
            [output for sublist in output_split for output in sublist])

        y = output_flatten[:self.seq_len]
        return y

    def forward(self, x):
        for i, (cell, rate) in enumerate(zip(self.cells, self.dilation)):
            x = self._forward(x, cell, rate)
        return x
