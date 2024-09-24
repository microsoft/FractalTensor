import torch
import typing
from typing import Tuple

import kaleido
from kaleido import Tensor
from kaleido import FractalTensor
from kaleido import TensorStorage

from kaleido import FractalTensorStorage

# ============= hyper parameters
vocab_size = 5000

# seq_len = 50
# batch_size = 16
# model_dim = 512
# ff_inner_dim = 2048
# depth = 6

# NOTE: small tensor size to make unittest run fast.
batch_size = 4
seq_len = 17
model_dim = 64
ff_inner_dim = 16
depth = 2

n_heads = 8
head_dim = int(model_dim / n_heads)
encoder_depth = 6

drop_rate = 0.5

# device = 'cpu'
device = 'cuda'


def create_before_attn_proj(shape=(model_dim, head_dim)):
    x = FractalTensor(TensorStorage(shape, kaleido.float32, device=device))
    x.indices = list(range(n_heads))
    x.initialize(torch.rand, *x.flatten_shape, device=device)
    return x


def atten_param():
    qkv_projs = FractalTensor.from_fractaltensors(*[
        create_before_attn_proj(),
        create_before_attn_proj(),
        create_before_attn_proj()
    ])

    layer_norm_scale = Tensor((1, model_dim), kaleido.float32, device=device)
    layer_norm_scale.initialize(
        torch.rand, *layer_norm_scale.shape, device=device)

    layer_norm_bias = Tensor((1, model_dim), kaleido.float32, device=device)
    layer_norm_bias.initialize(
        torch.rand, *layer_norm_bias.shape, device=device)

    ff_mat1 = Tensor((model_dim, ff_inner_dim), kaleido.float32, device=device)
    ff_mat1.initialize(torch.rand, *ff_mat1.shape, device=device)

    ff_bias1 = Tensor((1, ff_inner_dim), kaleido.float32, device=device)
    ff_bias1.initialize(torch.rand, *ff_bias1.shape, device=device)

    ff_mat2 = Tensor((ff_inner_dim, model_dim), kaleido.float32, device=device)
    ff_mat2.initialize(torch.rand, *ff_mat2.shape, device=device)

    ff_bias2 = Tensor((1, model_dim), kaleido.float32, device=device)
    ff_bias2.initialize(torch.rand, *ff_bias2.shape, device=device)

    return {
        'qkv_projs': qkv_projs,
        'layer_norm_scale': layer_norm_scale,
        'layer_norm_bias': layer_norm_bias,
        'ff_mat1': ff_mat1,
        'ff_bias1': ff_bias1,
        'ff_mat2': ff_mat2,
        'ff_bias2': ff_bias2
    }


def create_param(hidden_sizes: Tuple[int], num_heads: int,
                 device: str = 'cpu'):
    xs = FractalTensor(
        TensorStorage(hidden_sizes, kaleido.float32, device=device))
    xs.indices = list(range(num_heads))
    xs.initialize(torch.rand, *xs.flatten_shape, device=device)
    return xs


def create_input(hidden_dim: int,
                 seq_len: int,
                 batch_size: int,
                 device: str = 'cpu'):
    # depth-1: batch_size
    # depth-2: seq_length
    # depth-3: head_dim
    xss = FractalTensor(
        FractalTensorStorage(
            TensorStorage((1, hidden_dim), kaleido.float32, device=device)))
    indices = [list(range(seq_len)) for _ in range(batch_size)]
    xss.indices = indices
    xss.initialize(torch.rand, *xss.flatten_shape, device=device)
    return xss


def create_proj(hidden: int,
                num_heads: int,
                head_dim: int,
                device: str = 'cpu'):
    xss = FractalTensor(
        FractalTensorStorage(
            TensorStorage((1, hidden), kaleido.float32, device=device)))
    indices = [list(range(head_dim)) for _ in range(num_heads)]
    xss.indices = indices
    xss.initialize(torch.rand, *xss.flatten_shape, device=device)
    return xss


def create_proj_os(hidden: int,
                   num_heads: int,
                   head_dim: int,
                   device: str = 'cpu'):
    xs = FractalTensor(
        TensorStorage((head_dim, hidden), kaleido.float32, device=device))
    indices = list(range(num_heads))
    xs.indices = indices
    xs.initialize(torch.rand, *xs.flatten_shape, device=device)
    return xs
