import kaleido
from kaleido import Tensor

import torch

from examples.utils import gen_dataset

# ============= hyper parameters
device = 'cpu'
# device = 'cuda'

batch_size = 7
vocab_size = 5000
hidden_dim = 512

src_emb = Tensor((vocab_size, hidden_dim), kaleido.float32, device=device)
src_emb.initialize(torch.rand, *src_emb.shape, device=device)

trg_emb = Tensor((vocab_size, hidden_dim), kaleido.float32, device=device)
trg_emb.initialize(torch.rand, *trg_emb.shape, device=device)


def create_cell_param():
    W = Tensor((hidden_dim, hidden_dim), kaleido.float32, device=device)
    W.initialize(torch.rand, *W.shape, device=device)

    U = Tensor((hidden_dim, hidden_dim), kaleido.float32, device=device)
    U.initialize(torch.rand, *U.shape, device=device)

    b = Tensor((1, hidden_dim), kaleido.float32, device=device)
    b.initialize(torch.rand, *b.shape, device=device)
    return {'W': W, 'U': U, 'b': b}


src_params = create_cell_param()
trg_params = create_cell_param()

encoder_proj = Tensor((hidden_dim, 1), kaleido.float32, device=device)
encoder_proj.initialize(torch.rand, *encoder_proj.shape, device=device)

decoder_proj = Tensor((hidden_dim, 1), kaleido.float32, device=device)
decoder_proj.initialize(torch.rand, *decoder_proj.shape, device=device)

attn_params = (encoder_proj, decoder_proj)

src_words = gen_dataset(batch_size, vocab_size)
trg_words = gen_dataset(batch_size, vocab_size)
