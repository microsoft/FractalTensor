import torch

import kaleido
from kaleido import FractalTensor
from kaleido import TensorStorage

from kaleido import FractalTensorStorage


def create_input(head_dim: int,
                 num_heads: int,
                 seq_len: int,
                 batch_size: int,
                 block_dim: int,
                 device: str = 'cpu'):
    # depth-1: batch_size
    # depth-2: num_heads
    # depth-3: block_num = seq_length / block_dim
    xsss = FractalTensor(
        FractalTensorStorage(
            FractalTensorStorage(
                TensorStorage(
                    (block_dim, head_dim), kaleido.float32, device=device))))
    indices = []
    block_num = int(seq_len / block_dim)
    for _ in range(batch_size):
        indices.append([list(range(block_num)) for _ in range(num_heads)])
    xsss.indices = indices
    xsss.initialize(torch.rand, *xsss.flatten_shape, device=device)
    return xsss
