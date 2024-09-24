import context

from typing import Tuple, NamedTuple

import kaleido
from kaleido import Tensor, FractalTensor, StaticList, Iterative
from kaleido import operations as ops
from kaleido.parser.tests.utils import print_ast
from kaleido.parser.plot import PlotProgram

from grid_rnn_utils import *

ctx = kaleido.Context()


@kaleido.params(ctx)
class CellParams(NamedTuple):
    i2h: Tensor['256, 128', float, 'cpu']
    h2h: Tensor['128, 128', float, 'cpu']
    bias: Tensor['1, 128', float, 'cpu']


@kaleido.params(ctx)
class BlockParams(NamedTuple):
    x_direction: CellParams
    y_direction: CellParams


@kaleido.params(ctx)
class ModelParams(NamedTuple):
    src_emb: Tensor['5000, 128', float, 'cpu']
    trg_emb: Tensor['5000, 128', float, 'cpu']

    stacked_params: StaticList[BlockParams, '3']


# @kaleido.function(ctx)
def vanilla_cell(state: Tensor['1, 128', float, 'cpu'],
                 cur: Tensor['1, 256', float, 'cpu'],
                 i2h: Tensor['256, 128', float, 'cpu'],
                 h2h: Tensor['128, 128', float, 'cpu'],
                 bias: Tensor['1, 128', float, 'cpu']
                 ) -> Tensor['1, 128', float, 'cpu']:
    h = ops.tanh(cur @ i2h + state @ h2h + bias)
    return h


# @kaleido.function(ctx)
def grid_cell(
        state: Tuple[Tensor['1, 128', float, 'cpu'], Tensor['1, 128', float,
                                                            'cpu']],
        cur_input: Tuple[Tensor['1, 128', float, 'cpu'], Tuple[Tensor[
            '1, 128', float, 'cpu'], Tensor['1, 128', float, 'cpu']]],
        block_params: BlockParams
) -> Tuple[Tensor['1, 128', float, 'cpu'], Tensor['1, 128', float, 'cpu']]:
    _, state_y = state  # evaluations from the last execution instance.
    # unpack tuple elements, inputs to the current execution instance.
    state_x, (x_t, y_t) = cur_input

    # unpack tuple elements, get learnable parameters
    rnn_param_x, rnn_param_y = block_params

    s = ops.cat([state_x, state_y], dim=1)
    h_x = vanilla_cell(x_t, s, *rnn_param_x)
    h_y = vanilla_cell(y_t, s, *rnn_param_y)
    return h_x, h_y


# @kaleido.function(ctx)
def direction_y(
        state: Tuple[FractalTensor[Tensor['1, 128', float, 'cpu']],
                     FractalTensor[Tensor['1, 128', float, 'cpu']]],
        cur_input: Tuple[FractalTensor[Tensor['1, 128', float, 'cpu']],
                         FractalTensor[Tensor['1, 128', float, 'cpu']]],
        block_params: BlockParams
) -> Tuple[FractalTensor[Tensor['1, 128', float, 'cpu']], FractalTensor[Tensor[
        '1, 128', float, 'cpu']]]:
    state_xs, _ = state

    zero = ops.zeros(shape=(1, 128), device='cpu')
    ys = ops.scan(
        lambda s, x: grid_cell(s, x, block_params),
        ops.zip(state_xs, ops.zip(*cur_input)),
        initializer=(zero, zero))
    return ys


# @kaleido.function(ctx)
def direction_x(state: Tuple[FractalTensor[Tensor['1, 128', float, 'cpu']],
                             FractalTensor[Tensor['1, 128', float, 'cpu']]],
                block_params: BlockParams
                ) -> Tuple[FractalTensor[Tensor['1, 128', float, 'cpu']],
                           FractalTensor[Tensor['1, 128', float, 'cpu']]]:
    # len(state[0][0]) is the length of source language sequence
    zeros = ops.repeat(
        ops.zeros(shape=(1, 128), device='cpu'), len(state[0][0]))
    ys1, ys2 = ops.scan(
        lambda s, x: direction_y(s, x, block_params),
        state,
        initializer=(zeros, zeros))
    ys = ops.zip(ys1, ys2)
    return ys


# @kaleido.function(ctx)
def stacked_grid_rnns(
        src_encs: FractalTensor[Tensor['1, 128', float, 'cpu']],
        trg_encs: FractalTensor[Tensor['1, 128', float, 'cpu']],
        stacked_params: StaticList[BlockParams, '3']
) -> Tuple[FractalTensor[FractalTensor[Tensor['1, 128', float, 'cpu']]],
           FractalTensor[FractalTensor[Tensor['1, 128', float, 'cpu']]]]:
    yss1, yss2 = ops.fold(
        lambda inputs, params: direction_x(inputs, params),
        stacked_params,
        initializer=ops.zip(*ops.product(src_encs, trg_encs)))
    yss = ops.zip(yss1, yss2)
    return yss


# @kaleido.function(ctx)
def model(
        src_batch: FractalTensor[FractalTensor[Tensor['1,', int, 'cpu']]],
        trg_batch: FractalTensor[FractalTensor[Tensor['1,', int, 'cpu']]],
        params: ModelParams) -> Tuple[FractalTensor[FractalTensor[
            FractalTensor[Tensor['1, 128', float, 'cpu']]]], FractalTensor[
                FractalTensor[FractalTensor[Tensor['1, 128', float, 'cpu']]]]]:
    src_encs = ops.map(lambda words:ops.map(lambda word:
            ops.index(ops.slices(params.src_emb, dim=0), word), words),
            src_batch)
    trg_encs = ops.map(lambda words:ops.map(lambda word:
            ops.index(ops.slices(params.trg_emb, dim=0), word), words),
            trg_batch)

    # data parallelism in a mini-batch
    ysss = ops.map(lambda x: stacked_grid_rnns(*x, params.stacked_params),
                   ops.zip(src_encs, trg_encs))
    return ysss


# block = ctx.peek().ir_block
# p = PlotProgram()
# p.plot(block)

if __name__ == '__main__':
    stacked_params = Iterative.make_iterative(*[
        BlockParams(
            x_direction=CellParams(**create_cell()),
            y_direction=CellParams(**create_cell())) for _ in range(depth)
    ])

    params = ModelParams(
        src_emb=src_emb, trg_emb=trg_emb, stacked_params=stacked_params)

    xss, yss = model(src_words, trg_words, params)
