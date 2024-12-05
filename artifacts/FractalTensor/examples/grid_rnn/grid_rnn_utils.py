import random

import torch

import kaleido
from kaleido import Tensor
from examples.utils import gen_dataset

device = 'cpu'
# device = 'cuda'

depth = 3
batch_size = 16
vocab_size = 5000
hidden_dim = 128

MIN_LEN = 3
MAX_LEN = 7

src_words = gen_dataset(batch_size, vocab_size, device=device)
trg_words = gen_dataset(batch_size, vocab_size, device=device)

src_emb = Tensor((vocab_size, hidden_dim), kaleido.float32, device=device)
src_emb.initialize(torch.rand, *src_emb.shape, device=device)

trg_emb = Tensor((vocab_size, hidden_dim), kaleido.float32, device=device)
trg_emb.initialize(torch.rand, *trg_emb.shape, device=device)


def create_cell():
    i2h = Tensor((2 * hidden_dim, hidden_dim), kaleido.float32, device=device)
    i2h.initialize(torch.rand, *i2h.shape, device=device)

    h2h = Tensor((hidden_dim, hidden_dim), kaleido.float32, device=device)
    h2h.initialize(torch.rand, *h2h.shape, device=device)

    bias = Tensor((1, hidden_dim), kaleido.float32, device=device)
    bias.initialize(torch.rand, *bias.shape, device=device)
    return {'i2h': i2h, 'h2h': h2h, 'bias': bias}
