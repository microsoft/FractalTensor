import context

from typing import Tuple, NamedTuple

import kaleido
from kaleido import Tensor, FractalTensor
from kaleido import operations as ops
from kaleido.parser.plot import PlotProgram

from rnn_attention_utils import *

ctx = kaleido.Context()


@kaleido.params(ctx)
class CellParams(NamedTuple):
    W: Tensor['512, 512', float, 'cpu']
    U: Tensor['512, 512', float, 'cpu']
    b: Tensor['512, 512', float, 'cpu']


@kaleido.params(ctx)
class ModelParams(NamedTuple):
    src_emb: Tensor['5000, 512', float, 'cpu']
    trg_emb: Tensor['5000, 512', float, 'cpu']

    src_params: CellParams
    trg_params: CellParams

    attn_params: Tuple[Tensor['512, 1', float, 'cpu'], Tensor['512, 1', float,
                                                              'cpu']]


# @kaleido.function(ctx)
def cell(state: Tensor['1, 512', float, 'cpu'],
         hidden: Tensor['1, 512', float, 'cpu'],
         i2h: Tensor['512, 512', float, 'cpu'],
         h2h: Tensor['512, 512', float, 'cpu'],
         bias: Tensor['1, 512', float, 'cpu']
         ) -> Tensor['1, 512', float, 'cpu']:
    h = ops.tanh(hidden @ i2h + state @ h2h + bias)
    return h


# @kaleido.function(ctx)
def attn_func(state: Tensor['1, 512', float, 'cpu'],
              cur: Tensor['1, 512', float, 'cpu'],
              encoders: FractalTensor[Tensor['1, 512', float, 'cpu']],
              trg_params: CellParams) -> Tensor['1, 512', float, 'cpu']:

    decoder = cell(state, cur, *trg_params)

    scores = ops.map(lambda x: x @ encoder_proj + decoder @ decoder_proj,
                     encoders)
    weights = ops.slices(ops.softmax(ops.flatten(scores)), dim=0)

    v = ops.reduce(
        lambda s, x: ops.add(s, x),
        ops.map(lambda x: ops.scale(*x), ops.zip(weights, encoders)))
    return v


# @kaleido.function(ctx)
def attn_layer(srcs: FractalTensor[Tensor['1, 512', float, 'cpu']],
               trgs: FractalTensor[Tensor['1, 512', float, 'cpu']],
               trg_params: CellParams
               ) -> FractalTensor[Tensor['1, 512', float, 'cpu']]:
    attn = ops.scan(
        lambda s, x: attn_func(s, x, srcs, trg_params),
        trgs,
        initializer=ops.zeros(shape=(1, 512), device='cpu'))
    return attn


# @kaleido.function(ctx)
def model(src_words: FractalTensor[FractalTensor[Tensor['1,', int, 'cpu']]],
          trg_words: FractalTensor[FractalTensor[Tensor['1,', int, 'cpu']]],
          params: ModelParams
          ) -> FractalTensor[FractalTensor[Tensor['1, 512', float, 'cpu']]]:
    src_embs = ops.map(lambda xs: ops.map(lambda x:
        ops.index(ops.slices(params.src_emb, dim=0), x),
        xs), src_words)

    trg_embs = ops.map(lambda xs: ops.map(lambda x:
        ops.index(ops.slices(params.trg_emb, dim=0), x),
        xs), trg_words)

    src_enc_outs = ops.map(lambda xs: ops.scan(
        lambda s, x: cell(s, x, *params.src_params),
        xs,
        initializer=ops.zeros(shape=(1,512), device='cpu')), src_embs)

    yss = ops.map(lambda x: attn_layer(*x, params.trg_params),
                  ops.zip(src_enc_outs, trg_embs))
    return yss


# block = ctx.peek().ir_block
# p = PlotProgram()
# p.plot(block)

if __name__ == '__main__':
    params = ModelParams(
        src_emb=src_emb,
        trg_emb=trg_emb,
        src_params=CellParams(**create_cell_param()),
        trg_params=CellParams(**create_cell_param()),
        attn_params=attn_params)
    out = model(src_words, trg_words, params)
