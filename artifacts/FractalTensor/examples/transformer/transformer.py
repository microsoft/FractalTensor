import context
from typing import Tuple
from typing import NamedTuple

import torch
import kaleido
from kaleido import StaticList
from kaleido import Iterative
from kaleido import operations as ops

from examples.utils import gen_equal_length_seqs
from transformer_utils import *

ctx = kaleido.Context()


@kaleido.params(ctx)
class BlockParams(NamedTuple):
    qkv_projs: FractalTensor[FractalTensor[Tensor['64, 8', float, 'cpu']]]
    layer_norm_scale: Tensor['1, 64', float, 'cpu']
    layer_norm_bias: Tensor['1, 64', float, 'cpu']
    ff_mat1: Tensor['64, 16', float, 'cpu']
    ff_bias1: Tensor['1, 16', float, 'cpu']
    ff_mat2: Tensor['16, 64', float, 'cpu']
    ff_bias2: Tensor['1, 64', float, 'cpu']


@kaleido.params(ctx)
class ModelParams(NamedTuple):
    embedding: Tensor['1, 64', float, 'cpu']

    # NOTE: Annotated can be used in Python3.9
    # from typing import Annotated
    # block_params: Annotated[BlockParams, 2]
    block_params: StaticList[BlockParams, '2']


# @kaleido.function(ctx)
def Gelu(x: Tensor['1, 16', float, 'cpu']) -> Tensor['1, 16', float, 'cpu']:
    v = 0.044715 * ops.pow(x, 3)
    v = 0.5 * x * (1. + ops.tanh(2.5059928172283334 * (x + v)))
    return v


# @kaleido.function(ctx)
def single_heads_attn(qs: FractalTensor[Tensor['1, 8', float, 'cpu']],
                      ks: FractalTensor[Tensor['1, 8', float, 'cpu']],
                      vs: FractalTensor[Tensor['1, 8', float, 'cpu']]
                      ) -> FractalTensor[Tensor['1, 8', float, 'cpu']]:
    # attention for a single head.
    unnorm_score = ops.map(lambda q: ops.map(lambda k: ops.dot(k, q) / 8., ks),
                           qs)

    score = ops.map(lambda s: ops.slices(ops.softmax(ops.stack(s)), dim=0),
                    unnorm_score)

    weighted_vs = ops.map(
        lambda s: ops.map(lambda x: ops.scale(*x), ops.zip(s, vs)), score)

    y = ops.map(lambda vs: ops.reduce(ops.add, vs), weighted_vs)
    return y


# @kaleido.function(ctx)
def change_layout(
        xss: FractalTensor[FractalTensor[Tensor['1, 8', float, 'cpu']]]
) -> FractalTensor[Tensor['1, 64', float, 'cpu']]:
    v = ops.flatten(xss)
    v = ops.reshape(v, [n_heads, -1, head_dim])
    v = ops.permute(v, [1, 0, 2])
    v = ops.slices(ops.reshape(v, [-1, model_dim]), dim=0)
    return v


# @kaleido.function(ctx)
def ffn(x: Tensor['1, 64', float, 'cpu'],
        block_param: BlockParams) -> Tensor['1, 64', float, 'cpu']:
    x = x @ block_param.ff_mat1 + block_param.ff_bias1
    x = Gelu(x)
    # x = ops.relu(x)
    x = ops.dropout(x, drop_rate=drop_rate)
    x = x @ block_param.ff_mat2 + block_param.ff_bias2
    return x


# @kaleido.function(ctx)
def encoder_block(
        emb: FractalTensor[FractalTensor[Tensor['1, 64', float, 'cpu']]],
        pos_enc: FractalTensor[FractalTensor[Tensor['1, 64', float, 'cpu']]],
        block_param: BlockParams
) -> Tuple[FractalTensor[FractalTensor[Tensor['1, 64', float, 'cpu']]],
           FractalTensor[FractalTensor[Tensor['1, 64', float, 'cpu']]]]:
    pre_attn_projs = ops.map(lambda params: ops.map(
            lambda xs: ops.map(lambda p: ops.map(
            lambda x: x @ p, xs), params), emb), block_param.qkv_projs)

    # scaled dot product attention.
    encodings = ops.map(
        lambda xss: ops.map(lambda xs: single_heads_attn(*xs), ops.zip(*xss)),
        ops.zip(*ops.unbind(pre_attn_projs)))

    # change layout
    encodings = ops.map(lambda xss: change_layout(xss), encodings)

    # add the positional encodings
    encodings = ops.map(
        lambda xs: ops.map(lambda x: ops.add(*x), ops.zip(*xs)),
        ops.zip(encodings, pos_enc))

    # layer norm
    normed_encoding = ops.map(lambda xs: ops.map(lambda x:
        ops.layer_norm(x, w=block_param.layer_norm_scale,
            b=block_param.layer_norm_bias), xs), encodings)

    v = ops.map(lambda xs: ops.map(lambda x: ffn(x, block_param), xs),
                normed_encoding)
    return v, pos_enc


# @kaleido.function(ctx)
# TODO(ying): arguments of this function should be processed into
# literal constants
def sinusoidal_embeddings(seq_len: int, dim: int, batch_size: int, device: str
                          ) -> FractalTensor[Tensor['1, 64', float, 'cpu']]:
    inv_freq = 1. / (10000**(
        ops.arange(0, dim, 2, dtype=kaleido.float32, device=device) / dim))
    t = ops.arange(seq_len, dtype=kaleido.float32, device=device)
    freqs = ops.outer(t, inv_freq)

    first_half = ops.sin(freqs)
    second_half = ops.cos(freqs)

    v = ops.stack([first_half, second_half], dim=1)
    v = ops.permute(v, [0, 2, 1])
    v = ops.slices(ops.reshape(v, [v.shape[0], -1]), dim=0)
    v = ops.repeat(v, batch_size)
    return v


# NOTE: use hash mark # to write comments. Do not use multiline strings
# inside a set of triple quotes. Paser does not handle the latter.
# @kaleido.function(ctx)
def model(seq_batch: FractalTensor[FractalTensor[Tensor['1,', int, 'cpu']]],
          params: ModelParams
          ) -> FractalTensor[FractalTensor[Tensor['1, 64', float, 'cpu']]]:

    # This implementation does not pad variable length sequences
    # into a same length.
    embs = ops.map(lambda words: ops.map(lambda word:
            ops.index(ops.slices(params.embedding, dim=0), word), words),
            seq_batch)

    # generate positional encodings.
    pos_encs = sinusoidal_embeddings(seq_len, model_dim, batch_size, device)

    # sum up word embedding and positional embedding.
    embs = ops.map(lambda xs: ops.map(lambda x: ops.add(*x), ops.zip(*xs)),
                   ops.zip(embs, pos_encs))

    yss = ops.fold(
        lambda state, param: encoder_block(*state, param),
        params.block_params,
        initializer=(embs, pos_encs)),

    return yss


if __name__ == '__main__':
    seq_batch = gen_equal_length_seqs(
        batch_size, vocab_size, seq_len, device=device)

    embedding = Tensor((vocab_size, model_dim), kaleido.float32, device=device)
    embedding.initialize(torch.rand, *embedding.shape, device=device)

    blocks = [BlockParams(**atten_param()) for _ in range(depth)]

    params = ModelParams(
        embedding=embedding, block_params=Iterative.make_iterative(*blocks))

    enc_vecs = model(seq_batch, params)
