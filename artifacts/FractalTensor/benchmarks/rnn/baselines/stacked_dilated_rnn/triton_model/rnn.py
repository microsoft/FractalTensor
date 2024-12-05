from typing import Tuple
from typing import List

import torch.jit as jit
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_ as init
from torch import Tensor

from time import time

from .op import *


class LSTMCell(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_size: int,
                 device: str,
                 dtype=torch.float16):
        super(LSTMCell, self).__init__()
        self.device = device
        self.dtype = dtype
        self.size = (input_size, hidden_size, batch_size)
        self.Wi = init(
            nn.Parameter(
                torch.empty(
                    [input_size, hidden_size], device=device, dtype=dtype)))
        self.Wf = init(
            nn.Parameter(
                torch.empty(
                    [input_size, hidden_size], device=device, dtype=dtype)))
        self.Wo = init(
            nn.Parameter(
                torch.empty(
                    [input_size, hidden_size], device=device, dtype=dtype)))
        self.Wg = init(
            nn.Parameter(
                torch.empty(
                    [input_size, hidden_size], device=device, dtype=dtype)))

        self.Ui = init(
            nn.Parameter(
                torch.empty(
                    [hidden_size, hidden_size], device=device, dtype=dtype)))
        self.Uf = init(
            nn.Parameter(
                torch.empty(
                    [hidden_size, hidden_size], device=device, dtype=dtype)))
        self.Uo = init(
            nn.Parameter(
                torch.empty(
                    [hidden_size, hidden_size], device=device, dtype=dtype)))
        self.Ug = init(
            nn.Parameter(
                torch.empty(
                    [hidden_size, hidden_size], device=device, dtype=dtype)))

        self.bi = nn.Parameter(
            torch.ones([hidden_size], device=device, dtype=dtype))
        self.bf = nn.Parameter(
            torch.ones([hidden_size], device=device, dtype=dtype))
        self.bo = nn.Parameter(
            torch.ones([hidden_size], device=device, dtype=dtype))
        self.bg = nn.Parameter(
            torch.ones([hidden_size], device=device, dtype=dtype))

    def forward(self, input: Tensor, state_prev: Tuple[Tensor, Tensor],
                state_now: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        h, c = LSTMscan(input, (self.Wi, self.Wf, self.Wo, self.Wg, self.Ui,
                                self.Uf, self.Uo, self.Ug),
                        (self.bi, self.bf, self.bo, self.bg), state_prev,
                        state_now, self.size, self.device, self.dtype)

        return h, c


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
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dilation = dilation
        self.h_resident = [
            torch.empty(
                [batch_size * dilation[i], hidden_size],
                device=device,
                dtype=dtype) for i in range(len(dilation))
        ]
        self.c_resident = [
            torch.empty(
                [batch_size * dilation[i], hidden_size],
                device=device,
                dtype=dtype) for i in range(len(dilation))
        ]
        self.cells = torch.nn.ModuleList([
            LSTMCell(input_size, hidden_size, batch_size * dilation[i], device,
                     dtype) for i in range(len(dilation))
        ])

    def _forward(self, iter, input_x, cell, rate):

        pad_num = (rate - (self.seq_len % rate)) % rate
        padding_data = torch.zeros(
            pad_num,
            self.batch_size,
            self.input_size,
            device=self.device,
            dtype=torch.float16)
        input_x = torch.cat((input_x, padding_data))

        dilated_input = torch.stack(
            tuple(
                map(lambda m: m.flatten(start_dim=0, end_dim=1),
                    input_x.split(rate))),
            dim=0)

        h = torch.zeros(
            (dilated_input.size(1), self.hidden_size),
            device=self.device,
            dtype=self.dtype)
        c = torch.zeros(
            (dilated_input.size(1), self.hidden_size),
            device=self.device,
            dtype=self.dtype)
        hs = []

        for i in range(dilated_input.size(0)):
            h, c = cell(dilated_input[i], (h, c),
                        (self.h_resident[iter], self.c_resident[iter]))
            hs.append(h)

        output = torch.stack(hs)

        output_split = [
            torch.split(item, self.batch_size) for item in torch.unbind(output)
        ]

        output_flatten = torch.stack(
            [output for sublist in output_split for output in sublist])
        xs = output_flatten[:self.seq_len]
        return xs

    def forward(self, x):
        for i, (cell, rate) in enumerate(zip(self.cells, self.dilation)):
            x = self._forward(i, x, cell, rate)
