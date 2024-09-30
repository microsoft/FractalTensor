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


class VanillaRNNCell(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 batch_size: int,
                 device: str,
                 grid_dim=2,
                 dtype=torch.float32):
        super(VanillaRNNCell, self).__init__()
        self.device = device
        self.dtype = dtype
        self.size = (hidden_size, batch_size, grid_dim)
        self.W = init(
            nn.Parameter(
                torch.empty(
                    [hidden_size, hidden_size], device=device, dtype=dtype)))

        self.U = init(
            nn.Parameter(
                torch.empty(
                    [hidden_size * grid_dim, hidden_size],
                    device=device,
                    dtype=dtype)))

        self.b = nn.Parameter(
            torch.zeros([hidden_size], device=device, dtype=dtype))

    def forward(
            self,
            x_t: Tensor,
            y_t: Tensor,
            state: Tensor,
            state_resident: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        h_x, h_y = Vanilla_scan((self.W, self.U), (x_t, y_t), self.b, state,
                                state_resident, self.size, self.device,
                                self.dtype)

        return h_x, h_y


class StackedGridModel(nn.Module):
    def __init__(self,
                 depth: int,
                 src_len: int,
                 trg_len: int,
                 batch_size: int,
                 hidden_size: int,
                 device: str,
                 dtype=torch.float32):
        super(StackedGridModel, self).__init__()
        self.depth = depth
        self.src_len = src_len
        self.trg_len = trg_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        self.h_output = torch.zeros(
            depth, src_len, trg_len, 2, batch_size, hidden_size, device=device)

        self.cells = torch.nn.ModuleList([
            VanillaRNNCell(hidden_size, batch_size, device, 2,
                           dtype).to(device) for _ in range(depth)
        ])

    def forward(self, src_array_batch: Tensor, trg_array_batch: Tensor):
        h_x_resident = torch.empty(
            [self.batch_size, self.hidden_size],
            device=self.device,
            dtype=self.dtype)
        h_y_resident = torch.empty(
            [self.batch_size, self.hidden_size],
            device=self.device,
            dtype=self.dtype)
        d = 0
        for m in self.cells:
            for i in range(0, self.src_len, 1):
                for j in range(0, self.trg_len, 1):
                    if d == 0:
                        x_t = src_array_batch[i]
                        y_t = trg_array_batch[j]
                    else:
                        x_t = self.h_output[d - 1][i][j][0]
                        y_t = self.h_output[d - 1][i][j][1]

                    if i == 0:
                        state_x = torch.zeros(
                            self.batch_size,
                            self.hidden_size,
                            device=self.device)
                    else:
                        state_x = self.h_output[d][i - 1][j][0]

                    if j == 0:
                        state_y = torch.zeros(
                            self.batch_size,
                            self.hidden_size,
                            device=self.device)
                    else:
                        state_y = self.h_output[d][i][j - 1][0]

                    state = torch.cat([state_x, state_y], dim=1)

                    h_x, h_y = m(x_t, y_t, state, (h_x_resident, h_y_resident))

                    self.h_output[d][i][j][0] = h_x
                    self.h_output[d][i][j][1] = h_y

            d += 1

        return self.h_output
