import context

import kaleido
from kaleido import Tensor
from kaleido import operations as ops

from flash_attention_utils import *


def attn_func(prev_maxes: Tensor['64, 1', float, 'cpu'],
              prev_sums: Tensor['64, 1', float, 'cpu'],
              prev_out: Tensor['64, 32', float, 'cpu'],
              query: Tensor['64, 32', float, 'cpu'],
              key: Tensor['64, 32', float, 'cpu'],
              value: Tensor['64, 32', float, 'cpu']
              ) -> Tensor['64, 32', float, 'cpu']:
    # ==============  softmax for the current block  ====================#
    attn_weights = query @ key.T  # q@K^T
    cur_maxes = ops.max(attn_weights, dim=-1, keepdim=True)  # m(x_cur)
    exp_weights = ops.exp(attn_weights - cur_maxes)  # f(x_cur)
    # unnoramlized attention score @ values
    exp_values = exp_weights @ value
    # move the normalization step to the very end of the attention computation
    cur_sums = ops.sum(exp_weights, dim=-1, keepdim=True)  # l(x_cur)

    # =======================    renormalization  ======================#
    new_maxes = ops.maximum(cur_maxes, prev_maxes)  # update m(x)
    # renormalization factor for the previous block
    renorm_prev = ops.exp(prev_maxes - new_maxes)
    # renormalization factor for the current block
    renorm_cur = ops.exp(cur_maxes - new_maxes)

    # update normalization factor l(x)
    new_sums = renorm_prev * prev_sums + renorm_cur * cur_sums

    o = (prev_out * prev_sums * renorm_prev +
         renorm_cur * exp_values) / new_sums

    return new_maxes, new_sums, o


def per_block_func(query: Tensor['64, 32', float, 'cpu'],
                   ks: FractalTensor[Tensor['64, 32', float, 'cpu']],
                   vs: FractalTensor[Tensor['64, 32', float, 'cpu']]
                   ) -> FractalTensor[Tensor['64, 32', float, 'cpu']]:
    init_sum = ops.zeros(shape=(64, 1), device='cpu')
    init_max = ops.full(
        shape=(64, 1), fill_value=-kaleido.float32.max, device='cpu')
    init_o = ops.zeros(shape=(64, 32), device='cpu')
    _, _, o = ops.reduce(
        lambda state, kvs: attn_func(*state, query, *kvs),
        ops.zip(ks, vs),
        initializer=(init_sum, init_max, init_o))
    return o


def per_head_func(
        qs: FractalTensor[Tensor['64, 32', float, 'cpu']],
        ks: FractalTensor[Tensor['64, 32', float, 'cpu']],
        vs: FractalTensor[Tensor['64, 32', float, 'cpu']]
) -> FractalTensor[FractalTensor[Tensor['64, 32', float, 'cpu']]]:
    # qs, ks, bs: query, key, value blocks, each block has a shape of [64, 32]
    # iterate over blocks in a query
    os = ops.map(lambda q: per_block_func(q, ks, vs), qs)
    return os


def flash_attention(
        qsss: FractalTensor[FractalTensor[FractalTensor[Tensor[
            '64, 32', float, 'cpu']]]], ksss: FractalTensor[FractalTensor[
                FractalTensor[Tensor['64, 32', float, 'cpu']]]],
        vsss: FractalTensor[FractalTensor[FractalTensor[Tensor[
            '64, 32', float, 'cpu']]]]) -> FractalTensor[FractalTensor[
                FractalTensor[Tensor['64, 32', float, 'cpu']]]]:
    # iterate over training samples and heads
    osss = ops.map(
        lambda xss: ops.map(lambda xs: per_head_func(*xs), ops.zip(*xss)),
        ops.zip(qsss, ksss, vsss))
    return osss


if __name__ == '__main__':
    # block_dim = 64, a long sequence is split into blocks, each block has block_dim tokens.
    # depth-1: batch_size
    # depth-2: num_heads
    # depth-3: block_num = length / block_dim
    qsss = create_input(
        head_dim=32, num_heads=8, block_dim=64, seq_len=1024, batch_size=2)
    ksss = create_input(
        head_dim=32, num_heads=8, block_dim=64, seq_len=1024, batch_size=2)
    vsss = create_input(
        head_dim=32, num_heads=8, block_dim=64, seq_len=1024, batch_size=2)

    osss = flash_attention(qsss, ksss, vsss)
