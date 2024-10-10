# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import context

from examples.stacked_rnn.rnn_utils import *
from kaleido.parser.plot import PlotProgram


@kaleido.function(ctx)
def gate_f(
        w: Tensor['512, 512', float, 'cpu'],
        u: Tensor['512, 512', float, 'cpu'], b: Tensor['1, 512', float, 'cpu'],
        x: Tensor['1, 512', float, 'cpu'],
        s: Tensor['1, 512', float, 'cpu']) -> Tensor['1, 512', float, 'cpu']:
    y = x @ w + s @ u + b
    return y


@kaleido.function(ctx)
def lstm_cell(
        h_prev: Tensor['1, 512', float, 'cpu'],
        c_prev: Tensor['1, 512', float, 'cpu'],
        input: Tensor['1, 512', float, 'cpu'],
        ws: FractalTensor[Tensor['512, 512', float, 'cpu']],
        us: FractalTensor[Tensor['512, 512', float, 'cpu']],
        bs: FractalTensor[Tensor['1, 512', float, 'cpu']]
) -> Tuple[Tensor['1, 512', float, 'cpu'], Tensor['1, 512', float, 'cpu']]:
    prev_gate = ops.map(lambda x: gate_f(*x, input, h_prev), ops.zip(
        ws, us, bs))
    gate = ops.map(lambda x: ops.sigmoid(x), prev_gate[:3])
    c = gate[1] * c_prev + gate[0] * ops.tanh(prev_gate[3])
    h = gate[2] * ops.tanh(c)
    return h, c


# @kaleido.function(ctx)
def batched_cell(hs_prev: FractalTensor[Tensor['1, 512', float, 'cpu']],
                 cs_prev: FractalTensor[Tensor['1, 512', float, 'cpu']],
                 xs: FractalTensor[Tensor['1, 512', float, 'cpu']],
                 ws: FractalTensor[Tensor['512, 512', float, 'cpu']],
                 us: FractalTensor[Tensor['512, 512', float, 'cpu']],
                 bs: FractalTensor[Tensor['1, 512', float, 'cpu']]
                 ) -> Tuple[FractalTensor[Tensor['1, 512', float, 'cpu']],
                            FractalTensor[Tensor['1, 512', float, 'cpu']]]:
    hs, cs = ops.map(lambda x: lstm_cell(*x, ws, us, bs),
                     ops.zip(hs_prev[0:xs.length], cs_prev[0:xs.length], xs))
    return hs, cs


# @kaleido.function(ctx)
def lstm_layer(
        xss: FractalTensor[FractalTensor[Tensor['1, 512', float, 'cpu']]],
        ws: FractalTensor[Tensor['512, 521', float, 'cpu']],
        us: FractalTensor[Tensor['512, 512', float, 'cpu']],
        bs: FractalTensor[Tensor['1, 512', float, 'cpu']]
) -> FractalTensor[Tensor['1, 512', float, 'cpu']]:
    init = ops.repeat(ops.zeros(shape=(1, 512), device='cpu'), xss[0].length)
    hss, css = ops.scan(
        lambda state, x: batched_cell(*state, x, ws, us, bs),
        xss,
        initializer=(init, init))
    return hss


# @kaleido.function(ctx)
def stacked_lstm(embs: FractalTensor[Tensor['1, 512', float, 'cpu']],
                 Wss: FractalTensor[Tensor['512, 512', float, 'cpu']],
                 Uss: FractalTensor[Tensor['512, 512', float, 'cpu']],
                 bss: FractalTensor[Tensor['1, 512', float, 'cpu']]
                 ) -> FractalTensor[Tensor['1, 512', float, 'cpu']]:
    rnn_outs = ops.fold(
        lambda s, param: lstm_layer(s, *param),
        ops.zip(Wss, Uss, bss),
        initializer=embs)

    return rnn_outs


block = ctx[-1].ir_block
block.propagate_storage()

p = PlotProgram()
p.plot(block)

if __name__ == '__main__':
    rnn_outs = stacked_lstm(embs, params.Wss, params.Uss, params.bss)
