import context

from typing import NamedTuple

import kaleido
from kaleido import operations as ops

from sparse_attention_utils import *

ctx = kaleido.Context()


@kaleido.params(ctx)
class AttnParams(NamedTuple):
    q_projs: FractalTensor[Tensor['1024, 64', float, 'cuda']]
    k_projs: FractalTensor[Tensor['1024, 64', float, 'cuda']]
    v_projs: FractalTensor[Tensor['1024, 64', float, 'cuda']]


# @kaleido.function(ctx)
def sparse_attn(query_token: Tensor['1, 64', float, 'cuda'],
                w_keys: FractalTensor[Tensor['1, 64', float, 'cuda']],
                r_keys: FractalTensor[Tensor['1, 64', float, 'cuda']],
                w_values: FractalTensor[Tensor['1, 64', float, 'cuda']],
                r_values: FractalTensor[Tensor['1, 64', float, 'cuda']]
                ) -> Tensor['1, 64', float, 'cuda']:
    ks = ops.flatten(w_keys.join(r_keys)).T
    vs = ops.flatten(w_values.join(r_values))

    attn_vecs = ops.softmax(query_token @ ks) @ vs
    return attn_vecs


# @kaleido.function(ctx)
def per_head_attn(query: FractalTensor[Tensor['1, 64', float, 'cuda']],
                  key: FractalTensor[Tensor['1, 64', float, 'cuda']],
                  value: FractalTensor[Tensor['1, 64', float, 'cuda']],
                  random_attn_pos: FractalTensor[Tensor['1, 3', float, 'cuda']]
                  ) -> FractalTensor[Tensor['1, 64', float, 'cuda']]:
    assert query.length == key.length == value.length

    windowed_key, windowed_value = ops.shifted_slide(
        ops.zip(key, value), window_size=5, dilation=2)
    random_key, random_value = ops.map(lambda ids: (key[ids], value[ids]),
                                       random_attn_pos)

    # global attention
    _, g_query, g_key, g_value = ops.filter(
        lambda x: (x[0] < 3 or x[0] >= query.length - 3),
        ops.enumerate(query, key, value))
    g_query = ops.flatten(g_query)
    g_key = ops.flatten(g_key)
    g_value = ops.flatten(g_value)
    v1 = ops.slices(ops.softmax(g_query @ g_key.T) @ g_value, dim=0)

    # random attention and windowed attention
    _, s_query, s_key, s_value = ops.filter(
        lambda x: x[0] >= 3 and x[0] < query.length - 3,
        ops.enumerate(query, windowed_key, windowed_value))
    v2 = ops.map(lambda x: sparse_attn(*x),
                 ops.zip(s_query, s_key, random_key, s_value, random_value))

    start, end = ops.split(v1, 2)
    attn = start.join(v2).join(end)
    return attn


# @kaleido.function(ctx)
def multihead_sparse_attention(
        queries: FractalTensor[Tensor['1, 1024', float, 'cuda']],
        keys: FractalTensor[Tensor['1, 1024', float, 'cuda']],
        values: FractalTensor[Tensor['1, 1024', float, 'cuda']],
        params: AttnParams, random_attn: FractalTensor[FractalTensor[
            FractalTensor[Tensor['1, 3', int, 'cuda']]]]) -> FractalTensor[
                FractalTensor[FractalTensor[Tensor['1, 64', float, 'cuda']]]]:
    # query_proj is a depth-3 FractalTensor.
    # depth-1: batch_size, depth 2: num_heads, depth 3: sequence_length
    # leaf node: row vectors with a size of [1, hidden_size // num_heads]
    query_proj = ops.map(  # batch_size
        lambda query: ops.map(  # num_heads
            lambda proj: ops.map(  # sequence_length
                lambda token: token @ proj, query),
            params.q_projs),
        queries)
    key_proj = ops.map(  # batch_size
        lambda key: ops.map(  # num_heads
            lambda proj: ops.map(  # sequence_length
                lambda token: token @ proj, key),
            params.k_projs),
        keys)
    value_proj = ops.map(  # batch_size
        lambda value: ops.map(  # num_heads
            lambda proj: ops.map(  # sequence_length
                lambda token: token @ proj, value),
            params.v_projs),
        values)

    v = ops.map(  # batch_size
        lambda xs: ops.map(  # num_heads
            lambda x: per_head_attn(*x), ops.zip(*xs)),
        ops.zip(query_proj, key_proj, value_proj, random_attn))
    return v


if __name__ == '__main__':
    batch_size = 2

    # seq_len = 4096
    # block_size = 64

    seq_len = 16
    block_size = 4

    block_num = seq_len // block_size

    model_dim = 1024
    n_heads = 16

    random_num = 3

    window_size = 5
    dilation = 2

    # q, k, v are all depth-2 FractalTensors
    # depth-1 is the batch_size, depth-2 is the sequence_length.
    q = create_input(batch_size, seq_len, model_dim)
    k = create_input(batch_size, seq_len, model_dim)
    v = create_input(batch_size, seq_len, model_dim)

    attn_params = AttnParams(
        q_projs=create_before_attn_proj(n_heads, model_dim),  # num_heads
        k_projs=create_before_attn_proj(n_heads,
                                        model_dim),  # per-head projection
        v_projs=create_before_attn_proj(n_heads, model_dim))

    random_attn_pos = gen_random_atten_indices(
        batch_size,
        n_heads,
        seq_len,
        seq_len - 6,  # filter global attention
        random_num)
    multihead_sparse_attention(q, k, v, attn_params, random_attn_pos)
