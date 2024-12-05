from typing import Tuple
from typing import List

import torch.jit as jit
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_ as init
from torch import Tensor

__all__ = [
    'small_model',
]

import torch
from torch import nn
from torch import Tensor
from typing import Tuple


def init(param):
    return nn.init.kaiming_normal_(
        param, mode='fan_out', nonlinearity='sigmoid')


class FineGrainedOpLSTMCell_v1(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dtype=torch.float16):
        super(FineGrainedOpLSTMCell_v1, self).__init__()
        # learnable parameters for input gate.
        self.Wi = nn.Parameter(
            init(torch.Tensor(input_size, hidden_size).half()))
        self.Ui = nn.Parameter(
            init(torch.Tensor(hidden_size, hidden_size).half()))
        self.bi = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))

        # learnable parameters for forget gate.
        self.Wf = nn.Parameter(
            init(torch.Tensor(input_size, hidden_size).half()))
        self.Uf = nn.Parameter(
            init(torch.Tensor(hidden_size, hidden_size).half()))
        self.bf = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))

        # learnable parameters for cell candidate.
        self.Wg = nn.Parameter(
            init(torch.Tensor(input_size, hidden_size).half()))
        self.Ug = nn.Parameter(
            init(torch.Tensor(hidden_size, hidden_size).half()))
        self.bg = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))

        # learnable parameters for output gate.
        self.Wo = nn.Parameter(
            init(torch.Tensor(input_size, hidden_size).half()))
        self.Uo = nn.Parameter(
            init(torch.Tensor(hidden_size, hidden_size).half()))
        self.bo = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))

    def forward(self, input: Tensor,
                state_prev: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        h_prev, c_prev = state_prev

        ig = torch.sigmoid(input @ self.Wi + h_prev @ self.Ui + self.bi)
        fg = torch.sigmoid(input @ self.Wf + h_prev @ self.Uf + self.bf)
        og = torch.sigmoid(input @ self.Wo + h_prev @ self.Uo + self.bo)
        c_candidate = torch.tanh(input @ self.Wg + h_prev @ self.Ug + self.bg)

        c = fg * c_prev + ig * c_candidate
        h = og * torch.tanh(c)
        return h, c


class FineGrainedOpLSTMCell_v2(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dtype=torch.float16):
        super(FineGrainedOpLSTMCell_v2, self).__init__()
        # learnable parameters for four gates.
        self.W = nn.Parameter(
            init(
                torch.zeros(
                    (input_size, hidden_size * 4), dtype=torch.float16)))
        self.U = nn.Parameter(
            init(
                torch.zeros(
                    (hidden_size, hidden_size * 4), dtype=torch.float16)))
        self.b = nn.Parameter(torch.ones(hidden_size * 4, dtype=torch.float16))

        self.hidden_size = hidden_size

    def forward(self, input: Tensor,
                state_prev: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        h_prev, c_prev = state_prev

        g = input @ self.W + h_prev @ self.U + self.b
        ig, fg, og, c_candidate = g.chunk(4, 1)

        ig = torch.sigmoid(ig)
        fg = torch.sigmoid(fg)
        og = torch.sigmoid(og)
        c_candidate = torch.tanh(c_candidate)

        c = fg * c_prev + ig * c_candidate
        h = og * torch.tanh(c)
        return h, c


class CuDNNLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dtype):
        super(CuDNNLSTM, self).__init__()

        # The model uses the nn.RNN module (and its sister modules nn.GRU
        # and nn.LSTM) which will automatically use the cuDNN backend
        # if run on CUDA with cuDNN installed.
        self.rnn_net = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=0.,
            batch_first=False,  # layout is: [length, batch, hidden]
            bidirectional=False,
            dtype=dtype)

    def forward(self, input):
        hiddens, _ = self.rnn_net(input)
        return hiddens


class StackedLSTM(nn.Module):
    '''Container module for a stacked RNN language model.

    Args:
        cell_type: str, the recurrent net to compute sentence representation.
        vocab_size: int, the size of word vocabulary.
        embedding_dim: int, the dimension of word embedding.
        rnn_hidden_dim: int, the dimension of RNN cell's hidden state.
        num_layers: int, the number of stacked RNN network.
    Returns:
        A Tuple of Tensor. The first element is the final output of the
        model before loss computation with a shape of
            [batch_size, seq_len, vocab_size].
        The second element is the hidden states of the RNN network with a shape
            [batch_size, seq_len, rnn_hidden_dim].
    '''

    def __init__(self,
                 batch_size: int,
                 max_seq_length: int,
                 cell_type: str,
                 hidden_size: int,
                 num_layers: int,
                 dtype=torch.float16):
        super(StackedLSTM, self).__init__()
        self.max_seq_length = max_seq_length

        self.register_buffer(
            'init_state', torch.zeros((batch_size, hidden_size), dtype=dtype))

        if cell_type == 'v1':
            self.cells = nn.ModuleList([
                FineGrainedOpLSTMCell_v1(
                    hidden_size, hidden_size, dtype=dtype)
                for i in range(num_layers)
            ])
        elif cell_type == 'v2':
            self.cells = nn.ModuleList([
                FineGrainedOpLSTMCell_v2(
                    hidden_size, hidden_size, dtype=dtype)
                for i in range(num_layers)
            ])
        else:
            raise ValueError(f'Unknown cell type {cell_type}.')

    def forward(self, input):
        '''Define forward computations of the RNNLM.

        Args:
            input: Tensor, its shape is [batch_size, seq_len], dtype is float16.

        Returns:
            A Tensor with a shape of [batch_size, seq_len, rnn_hidden_dim], dtype is float16.
        '''

        xs = input

        hiddens = torch.jit.annotate(List[Tensor], [])
        cells = torch.jit.annotate(List[Tensor], [])
        for rnn in self.cells:
            h: Tensor = self.init_state
            c: Tensor = self.init_state

            hs = torch.jit.annotate(List[Tensor], [])
            cs = torch.jit.annotate(List[Tensor], [])

            inputs = xs.unbind(0)
            for i in range(self.max_seq_length):
                h, c = rnn(inputs[i], (h, c))

                hs.append(h.half())  # Ensure the tensor is float16
                cs.append(c.half())  # Ensure the tensor is float16

            hs = torch.stack(hs)
            cs = torch.stack(cs)
            xs = hs

            hiddens.append(hs)
            cells.append(cs)
        return hiddens, cells


def small_model(cell_type,
                batch_size,
                max_seq_length,
                hidden_size,
                num_layers,
                dtype,
                states=False):
    if cell_type == 'cudnn_lstm':
        return CuDNNLSTM(
            hidden_size=hidden_size, num_layers=num_layers, dtype=dtype)
    elif cell_type == 'script_lstm':
        return ScriptLSTM(hidden_size, hidden_size, num_layers, states)
    else:
        return StackedLSTM(
            batch_size=batch_size,
            cell_type=cell_type,
            max_seq_length=max_seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dtype=dtype)
