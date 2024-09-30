from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import zeros_
from torch.nn.init import xavier_normal_ as init

__all__ = ['StackedGridModel']


class VanillaRNNCell(Module):
    def __init__(self, hidden_size, grid_dim=2):
        """
        Args:
            hidden_size(int): hidden dimension
            grid_dim(int): grid dimension
        """
        super(VanillaRNNCell, self).__init__()

        # learnable paramters
        self.W = Parameter(Tensor(hidden_size, hidden_size))
        self.U = Parameter(Tensor(hidden_size * grid_dim, hidden_size))
        self.b = Parameter(Tensor(1, hidden_size))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                init(p.data)
            else:
                zeros_(p.data)

    def forward(self, x_t: Tensor, y_t: Tensor,
                state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x_t(Tensor):
                the shape is (batch_size, hidden_size)
            y_t(Tensor):
                the shape is (batch_size, hidden_size)   
            state(Tensor):
                the shape is (batch_size, grid_dim * hidden_size)   
        Returns:
            (h_x, h_y): Tuple[Tensor, Tensor]
                h_x:
                    the shape is (batch_size, hidden_size)   
                h_y:
                    the shape is (batch_size, hidden_size)
        """
        temp = torch.mm(state, self.U) + self.b

        h_x = torch.tanh(torch.mm(x_t, self.W) + temp)
        h_y = torch.tanh(torch.mm(y_t, self.W) + temp)
        return h_x, h_y


class GridRNNNaive(Module):
    def __init__(self, depth: int, src_len: int, trg_len: int, batch_size: int,
                 hidden_size: int, device: str):
        """
        Args:
            depth(int): the number of stacked RNN layer
            src_len(int): source sequence length
            trg_len(int): target sequence length
            batch_size(int): the number of samples
            hidden_size(int): hidden dimension
        """
        super(GridRNNNaive, self).__init__()

        self.depth = depth
        self.src_len = src_len
        self.trg_len = trg_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device

        # We stack 2d GridLSTMs to get 3d GridLSTM.
        self.cells = torch.nn.ModuleList(
            [VanillaRNNCell(hidden_size).to(device) for _ in range(depth)])

        self.h_output = torch.zeros(
            self.depth,
            src_len,
            trg_len,
            2,
            batch_size,
            self.hidden_size,
            device=self.device)

    def forward(self, src_array_batch: Tensor, trg_array_batch: Tensor):
        """
        Args:
            src_array_batch(Tensor):
                the shape is (src_len, batch_size, hidden_size)
            trg_array_batch(Tensor):
                the shape is (trg_len, batch_size, hidden_size)        
        Returns:
            h_output(Tensor):
                the shape is (depth, src_len, trg_len, grid_dim, batch_size, hidden_size)
        """
        # dim 1: stack Grid LSTM Cell to form depth.
        d = 0
        for m in self.cells:
            # dim 2: iterate over source sequence length.
            for i in range(0, self.src_len, 1):
                # dim 3: iterate over target sequence length.
                for j in range(0, self.trg_len, 1):

                    # print("depth:", m, " src:", i, " trg:", j)
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

                    h_x, h_y = m(x_t, y_t, state)

                    self.h_output[d][i][j][0] = h_x
                    self.h_output[d][i][j][1] = h_y

            d += 1

        return self.h_output


class StackedGridModel(Module):
    def __init__(self, depth: int, src_len: int, trg_len: int, batch_size: int,
                 hidden_size: int, device: str, enable_jit: bool):
        """
        Args:
            depth(int): the number of stacked RNN layer
            src_len(int): source sequence length
            trg_len(int): target sequence length
            batch_size(int): the number of samples
            hidden_size(int): hidden dimension
            enable_jit(bool): whether to apply PyTorch JIT
        """
        super().__init__()

        self.depth = depth
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = device

        if enable_jit:
            self.m = torch.jit.script(
                GridRNNNaive(self.depth, src_len, trg_len, batch_size,
                             self.hidden_size, self.device)).to(self.device)
        else:
            self.m = GridRNNNaive(self.depth, src_len, trg_len, batch_size,
                                  self.hidden_size, self.device).to(
                                      self.device)

    def forward(self, source_input, target_input):
        """
        Args:
            src_array_batch(Tensor):
                the shape is (src_len, batch_size, hidden_size)
            trg_array_batch(Tensor):
                the shape is (trg_len, batch_size, hidden_size)        
        Returns:
            h_output(Tensor):
                the shape is (depth, src_len, trg_len, grid_dim, batch_size, hidden_size)
        """

        output = self.m(
            src_array_batch=source_input, trg_array_batch=target_input)

        return output
