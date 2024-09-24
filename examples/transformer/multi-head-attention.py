import context
import kaleido

from kaleido import Tensor
from kaleido import FractalTensor
from kaleido import operations as ops

from transformer_utils import create_input, create_proj, create_proj_os


def attn_func(query: Tensor['1, 32', float, 'cpu'],
              ks: FractalTensor[Tensor['1, 32', float, 'cpu']],
              vs: FractalTensor[Tensor['1, 32', float, 'cpu']]
              ) -> Tensor['1, 32', float, 'cpu']:
    dots = ops.map(lambda k: ops.dot(query, k), ks)
    max_value = ops.reduce(
        lambda s, x: ops.maximum(s, x),
        dots,
        initializer=ops.full((1, 1), -kaleido.float32.max))
    exps = ops.map(lambda x: ops.exp(x - max_value), dots)
    norm = ops.reduce(
        lambda s, x: s + x,
        exps,
        initializer=ops.zeros(shape=(1, 1), device='cpu'))
    vecs = ops.map(lambda xs: xs[0] / norm * xs[1], ops.zip(exps, vs))
    o = ops.reduce(
        lambda s, x: s + x,
        vecs,
        initializer=ops.zeros(shape=(1, 32), device='cpu'))
    return o


def per_query_func(
        qs: FractalTensor[Tensor['1, 32', float, 'cpu']],
        ks: FractalTensor[Tensor['1, 32', float, 'cpu']],
        vs: FractalTensor[Tensor['1, 32', float, 'cpu']],
) -> FractalTensor[Tensor['1, 32', float, 'cpu']]:
    # iterate over query token in a query sequence
    os = ops.map(lambda q: attn_func(q, ks, vs), qs)
    return os


def f0(x: Tensor['1, 256', float, 'cpu'],
       wxs: FractalTensor[Tensor['1, 256', float, 'cpu']]
       ) -> Tensor['1, 32', float, 'cpu']:
    v = ops.stack(ops.map(lambda wx: ops.dot(wx, x), wxs))
    return v


def f2(states: FractalTensor[Tensor['1, 256', float, 'cpu']],
       os: FractalTensor[Tensor['1, 256', float, 'cpu']],
       wo: Tensor['32, 256', float, 'cpu']
       ) -> FractalTensor[Tensor['1, 256', float, 'cpu']]:
    vs = ops.map(lambda xs: xs[0] + xs[1] @ wo, ops.zip(states, os))
    return vs


def f1(oss: FractalTensor[FractalTensor[Tensor['1, 256', float, 'cpu']]],
       wos: FractalTensor[Tensor['1, 256', float, 'cpu']]
       ) -> FractalTensor[Tensor['1, 256', float, 'cpu']]:
    # reduction over the num_head dimension
    vs = ops.reduce(
        lambda s, xs: f2(s, *xs),
        ops.zip(oss, wos),
        initializer=ops.repeat(
            ops.zeros(shape=(1, 256), device='cpu'), len(oss[0])))
    return vs


def mha(qss: FractalTensor[FractalTensor[Tensor['1, 256', float, 'cpu']]],
        kss: FractalTensor[FractalTensor[Tensor['1, 256', float, 'cpu']]],
        vss: FractalTensor[FractalTensor[Tensor['1, 256', float, 'cpu']]],
        wqss: FractalTensor[FractalTensor[Tensor['1, 256', float, 'cpu']]],
        wkss: FractalTensor[FractalTensor[Tensor['1, 256', float, 'cpu']]],
        wvss: FractalTensor[FractalTensor[Tensor['1, 256', float, 'cpu']]],
        wos: FractalTensor[Tensor['32, 256', float, 'cpu']]) -> FractalTensor[
            FractalTensor[FractalTensor[Tensor['1, 256', float, 'cpu']]]]:
    # depth-1: batch_size, depth-2: head_num, depth-3: length
    qsss = ops.map(
        lambda qs: ops.map(lambda wqs: ops.map(lambda q: f0(q, wqs), qs), wqss),
        qss)
    ksss = ops.map(
        lambda ks: ops.map(lambda wks: ops.map(lambda k: f0(k, wks), ks), wkss),
        kss)
    vsss = ops.map(
        lambda vs: ops.map(lambda wvs: ops.map(lambda v: f0(v, wvs), vs), wvss),
        vss)

    # depth-1: batch_size, depth-2: head_num, depth-3: length
    osss = ops.map(
        lambda xs: ops.map(lambda x: per_query_func(*x), ops.zip(*xs)),
        ops.zip(qsss, ksss, vsss))

    # depth-1: batch_size, depth-2: length
    oss = ops.map(lambda oss: f1(oss, wos), osss)

    return oss


if __name__ == '__main__':
    head_dim = 32
    num_heads = 8
    hidden_dim = head_dim * num_heads
    seq_len = 10
    batch_size = 3

    # block_dim = 64, a long sequence is split into blocks, each block has block_dim tokens.
    # depth-1: batch_size
    # depth-2: seq_len
    qss = create_input(
        hidden_dim=hidden_dim, seq_len=seq_len, batch_size=batch_size)
    kss = create_input(
        hidden_dim=hidden_dim, seq_len=seq_len, batch_size=batch_size)
    vss = create_input(
        hidden_dim=hidden_dim, seq_len=seq_len, batch_size=batch_size)

    # depth-1: head_num
    # depth-2: head_dim
    wqss = create_proj(hidden_dim, num_heads, head_dim)
    wkss = create_proj(hidden_dim, num_heads, head_dim)
    wvss = create_proj(hidden_dim, num_heads, head_dim)
    wos = create_proj_os(hidden_dim, num_heads, head_dim)

    oss = mha(qss, kss, vss, wqss, wkss, wvss, wos)
