import context

import kaleido
from kaleido import operations as ops

from sparse_attention_utils import *


def norm(g1: Tensor['32, 32', float, 'cuda'],
         w1: Tensor['32, 96', float, 'cuda'],
         g2: Tensor['32, 32', float, 'cuda']
         ) -> FractalTensor[Tensor['32, 32', float, 'cuda']]:
    v = ops.softmax(ops.cat((g1, w1, g2), 1), 1)
    v = ops.split(v, 5, 1)
    return v


def attn_func(qs: FractalTensor[Tensor['32, 512', float, 'cuda']],
              ks: FractalTensor[Tensor['32, 512', float, 'cuda']],
              vs: FractalTensor[Tensor['32, 512', float, 'cuda']]
              ) -> FractalTensor[Tensor['32, 512', float, 'cuda']]:
    # windowed attention and global attention
    # NOTE: Multiple heads and random attention are OMITTED for brevity.
    wks, wvs = ops.shifted_slide(ops.zip(ks, vs), window_size=3)
    wys = ops.map(lambda x: x[0] @ ops.flatten(x[1]).T,
                  ops.zip(qs[2:-2], wks[2:-2]))
    gys1 = ops.map(lambda x: x @ ks[0].T, qs[2:-2])  # left global attention
    gys2 = ops.map(lambda x: x @ ks[-1].T, qs[2:-2])  # right global attention

    normed_vecs = ops.map(lambda x: norm(*x), ops.zip(gys1, wys, gys2))

    gvs1 = ops.map(lambda x: x[0] @ vs[0], normed_vecs)
    gvs2 = ops.map(lambda x: x[-1] @ vs[-1], normed_vecs)

    wvs = ops.map(lambda x: ops.flatten(x[0][1:-1]).T @ ops.flatten(x[1]),
                  ops.zip(normed_vecs, wvs[2:-2]))
    vs = ops.map(lambda x: x[0] + x[1] + x[2], ops.zip(gvs1, gvs2, wvs))
    return vs


def bigbird(
        qss: FractalTensor[FractalTensor[Tensor['32, 512', float, 'cuda']]],
        kss: FractalTensor[FractalTensor[Tensor['32, 512', float, 'cuda']]],
        vss: FractalTensor[FractalTensor[Tensor['32, 512', float, 'cuda']]]
) -> FractalTensor[FractalTensor[Tensor['32, 512', float, 'cuda']]]:
    v = ops.map(lambda xs: attn_func(*xs), ops.zip(qss, kss, vss))
    return v


if __name__ == '__main__':
    batch_size = 16
    seq_len = 4096
    hidden = 512
    block_size = 32

    qss = create_blocked_input(batch_size, hidden, block_size, seq_len)
    kss = create_blocked_input(batch_size, hidden, block_size, seq_len)
    vss = create_blocked_input(batch_size, hidden, block_size, seq_len)

    bigbird(qss, kss, vss)
