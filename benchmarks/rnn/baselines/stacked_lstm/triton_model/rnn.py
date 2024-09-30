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

    def forward(
            self,
            input: Tensor,
            state_prev: Tuple[Tensor, Tensor],
            state_now: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:

        h, c = LSTMscan(input, (self.Wi, self.Wf, self.Wo, self.Wg, self.Ui,
                                self.Uf, self.Uo, self.Ug),
                        (self.bi, self.bf, self.bo, self.bg), state_prev,
                        state_now, self.size, self.device, self.dtype)

        return h, c


class StackedLSTM(nn.Module):
    def __init__(self,
                 batch_size: int,
                 max_seq_length: int,
                 hidden_size: int,
                 num_layers: int,
                 device: str,
                 dtype=torch.float16):
        super(StackedLSTM, self).__init__()
        self.max_seq_length = max_seq_length
        self.device = device
        self.dtype = dtype
        self.size = (batch_size, hidden_size)
        self.cells = torch.nn.ModuleList([
            LSTMCell(hidden_size, hidden_size, batch_size, device, dtype)
            for i in range(num_layers)
        ])

    def forward(self, input):
        xs = input
        batch_size, hidden_size = self.size
        h_resident = torch.empty(
            [batch_size, hidden_size], device=self.device, dtype=self.dtype)
        c_resident = torch.empty(
            [batch_size, hidden_size], device=self.device, dtype=self.dtype)
        hiddens = []
        cells = []
        for rnn in self.cells:

            h = torch.zeros(
                (batch_size, hidden_size),
                device=self.device,
                dtype=self.dtype)
            c = torch.zeros(
                (batch_size, hidden_size),
                device=self.device,
                dtype=self.dtype)

            hs = []
            cs = []

            inputs = xs.unbind(0)
            for i in range(self.max_seq_length):
                h, c = rnn(inputs[i], (h, c), (h_resident, c_resident))

                hs.append(h)
                cs.append(c)

            hs = torch.stack(hs)
            cs = torch.stack(cs)
            xs = hs

            hiddens.append(hs)
            cells.append(cs)

        return hiddens, cells
