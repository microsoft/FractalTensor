import torch

import random

import kaleido
from kaleido import Tensor
from kaleido import FractalTensor
from kaleido import TensorStorage
from kaleido import FractalTensorStorage

# device = 'cpu'
device = 'cuda'


def create_before_attn_proj(n_heads: int, model_dim: int):
    head_dim = model_dim // n_heads
    shape = (model_dim, head_dim)

    x = FractalTensor(TensorStorage(shape, kaleido.float32, device=device))
    x.indices = list(range(n_heads))
    x.initialize(torch.rand, *x.flatten_shape, device=device)
    return x


def create_input(batch_size: int, seq_len: int, model_dim: int):
    shape = (1, model_dim)
    x = FractalTensor(
        FractalTensorStorage(
            TensorStorage(shape, kaleido.float32, device=device)))
    x.indices = [list(range(seq_len)) for _ in range(batch_size)]
    x.initialize(torch.rand, *x.flatten_shape, device=device)
    return x


def create_blocked_input(batch_size: int,
                         hidden: int,
                         block_size: int,
                         seq_len: int,
                         device=device):
    block_num = seq_len // block_size

    x = FractalTensor(
        FractalTensorStorage(
            TensorStorage(
                (block_size, hidden), kaleido.float32, device=device)))
    x.indices = [list(range(block_num)) for _ in range(batch_size)]
    x.initialize(torch.rand, *x.flatten_shape, device=device)
    return x


def gen_random_atten_indices(
        batch_size: int, n_heads: int, seq_len: int, N: int, random_num: int
) -> FractalTensor[FractalTensor[FractalTensor[Tensor['1, 3', int, 'cuda']]]]:
    """
    NOTE, this is a fake implementation wher random attention positions are
    probably overlapped with windowed attention.
    """
    random_indices = []
    for i in range(batch_size):
        indices_head = []
        for j in range(n_heads):
            indices_seq = []
            for k in range(N):
                t = Tensor((1, random_num), kaleido.int32, device=device)
                t.data = torch.IntTensor(
                    random.sample(list(range(seq_len)), 3)).to(device)
                indices_seq.append(t)
            indices_head.append(FractalTensor.from_tensors(*indices_seq))
        random_indices.append(FractalTensor.from_fractaltensors(*indices_head))
    v = FractalTensor.from_fractaltensors(*random_indices)
    return v
