import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_ as init
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor

__all__ = [
    'LSTMCell',
    'LSTMLayer',
    'StackedLSTM',
]


class LSTMCell(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.Wi = init(nn.Parameter(torch.Tensor(input_size, hidden_size)))
        self.Ui = init(nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
        self.bi = nn.Parameter(torch.ones(hidden_size))

        self.Wf = init(nn.Parameter(torch.Tensor(input_size, hidden_size)))
        self.Uf = init(nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
        self.bf = nn.Parameter(torch.ones(hidden_size))

        self.Wg = init(nn.Parameter(torch.Tensor(input_size, hidden_size)))
        self.Ug = init(nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
        self.bg = nn.Parameter(torch.ones(hidden_size))

        self.Wo = init(nn.Parameter(torch.Tensor(input_size, hidden_size)))
        self.Uo = init(nn.Parameter(torch.Tensor(hidden_size, hidden_size)))
        self.bo = nn.Parameter(torch.ones(hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, h_prev: Tensor,
                c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        ig: Tensor = torch.sigmoid(input @ self.Wi + h_prev @ self.Ui +
                                   self.bi)

        fg: Tensor = torch.sigmoid(input @ self.Wf + h_prev @ self.Uf +
                                   self.bf)

        c_candidate: Tensor = torch.tanh(input @ self.Wg + h_prev @ self.Ug +
                                         self.bg)

        og: Tensor = torch.sigmoid(input @ self.Wo + h_prev @ self.Uo +
                                   self.bo)

        c = fg.mul(c_prev) + ig.mul(c_candidate)
        h = og.mul(torch.tanh(c))
        return h, c


class LSTMLayer(jit.ScriptModule):
    def __init__(self, *cell_args):
        super().__init__()
        self.cell = LSTMCell(*cell_args)

    @jit.script_method
    def forward(
            self,
            input: Tensor,
            h: Tensor,
            c: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        hiddens = []
        states = []
        for i in range(input.size(0)):
            h, c = self.cell(input[i], h, c)
            hiddens += [h]
            states += [h]
        return torch.stack(hiddens), torch.stack(states)


class StackedLSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer1 = LSTMLayer(input_size, hidden_size)
        self.layer2 = LSTMLayer(input_size, hidden_size)
        self.layer3 = LSTMLayer(input_size, hidden_size)

    @jit.script_method
    def forward(self, input: Tensor, h_inits: List[Tensor],
                c_inits: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        hiddens = jit.annotate(List[Tensor], [])
        cells = jit.annotate(List[Tensor], [])

        hs, cs = self.layer1(input, h_inits[0], c_inits[0])
        hiddens += [hs]
        cells += [cs]

        hs, cs = self.layer2(hs, h_inits[1], c_inits[1])
        hiddens += [hs]
        cells += [cs]

        hs, cs = self.layer3(hs, h_inits[2], c_inits[2])
        hiddens += [hs]
        cells += [cs]
        return hiddens, cells
