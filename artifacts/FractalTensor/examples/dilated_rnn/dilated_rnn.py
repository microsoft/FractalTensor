from typing import Tuple

import context

from examples.stacked_rnn.rnn_utils import *
from examples.stacked_rnn.stacked_rnn import lstm_cell
from examples.utils import gen_dataset


# @kaleido.function(ctx)
def dilated_layer(state: FractalTensor[Tensor['1, 512', float, 'cpu']],
                  itr: int,
                  Ws: FractalTensor[Tensor['512, 521', float, 'cpu']],
                  Us: FractalTensor[Tensor['512, 512', float, 'cpu']],
                  bs: FractalTensor[Tensor['1, 512', float, 'cpu']]
                  ) -> FractalTensor[Tensor['1, 512', float, 'cpu']]:
    zeros = ops.zeros(shape=(1, 512), device='cpu', dtype='float')
    h, _ =  ops.dilated_map(
        lambda xs: ops.scan(lambda s, x: lstm_cell(*s, x, Ws, Us, bs),
            xs, initializer=(zeros, zeros)),
        state,
        dilation=2**itr)
    return h


# @kaleido.function(ctx)
def model(batch_words: FractalTensor[FractalTensor[Tensor['1,', int, 'cpu']]],
          params: ModelParams
          ) -> FractalTensor[FractalTensor[Tensor['1, 512', float, 'cpu']]]:
    embs = ops.map(lambda words: ops.map(lambda word:
            ops.index(ops.slices(params.embedding, dim=0), word), words),
            batch_words)
    itrs = ops.enumerate(params.Wss, params.Uss, params.bss)
    rnn_outs = ops.map(lambda xs: ops.fold(lambda s, x: dilated_layer(s, *x),
        itrs, initializer=xs), embs)
    return rnn_outs


if __name__ == '__main__':
    xss = gen_dataset(batch_size, vocab_size)
    yss = model(xss, params)
