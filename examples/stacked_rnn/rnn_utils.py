from typing import Tuple

import random
import torch

from typing import NamedTuple

import kaleido
from kaleido import Tensor, FractalTensor
from kaleido import TensorStorage, FractalTensorStorage
from kaleido import operations as ops

from examples.utils import gen_dataset
from examples.utils import gen_dataset_time_major

# ============= hyper parameters
batch_size = 7
vocab_size = 5000
hidden_dim = 512

depth = 3

device = 'cpu'

# device = 'cuda'


def create_stacked_params(shape, depth):
    # projection matrix for 3 gates and cell candidate.
    x = FractalTensor(
        FractalTensorStorage(
            TensorStorage(shape, kaleido.float32, device=device)))
    x.indices = [list(range(4)) for _ in range(depth)]

    x.initialize(torch.rand, *x.flatten_shape, device=device)
    return x


def create_params():
    embedding = Tensor(
        (vocab_size, hidden_dim), kaleido.float32, device=device)
    embedding.initialize(torch.rand, *embedding.shape, device=device)

    prev_softmax_proj = Tensor(
        (hidden_dim, vocab_size), kaleido.float32, device=device)
    prev_softmax_proj.initialize(
        torch.rand, *prev_softmax_proj.shape, device=device)

    Wss = create_stacked_params([hidden_dim, hidden_dim], depth)
    Uss = create_stacked_params([hidden_dim, hidden_dim], depth)
    bss = create_stacked_params([1, hidden_dim], depth)

    return {
        'embedding': embedding,
        'prev_softmax_proj': prev_softmax_proj,
        'Wss': Wss,
        'Uss': Uss,
        'bss': bss
    }


ctx = kaleido.Context()


@kaleido.params(ctx)
class ModelParams(NamedTuple):
    embedding: Tensor['5000, 64', float, 'cpu']
    prev_softmax_proj: Tensor['512, 5000', float, 'cpu']

    Wss: FractalTensor[FractalTensor[Tensor['512, 512', float, 'cpu']]]
    Uss: FractalTensor[FractalTensor[Tensor['512, 512', float, 'cpu']]]
    bss: FractalTensor[FractalTensor[Tensor['1, 512', float, 'cpu']]]


params = ModelParams(**create_params())

batch_words = gen_dataset(batch_size, vocab_size, device=device)
batched_embs = ops.map(lambda words: ops.map(lambda word:
        ops.index(ops.slices(params.embedding, dim=0), word), words),
        batch_words)

words = gen_dataset_time_major(batch_size, vocab_size, device=device)
embs = ops.map(lambda ws: ops.map(lambda word:
            ops.index(ops.slices(params.embedding, dim=0), word), ws),
       words)
