# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import List, Tuple

import torch

import kaleido
from kaleido import FractalTensor, FractalTensorStorage, Tensor, TensorStorage
from kaleido import operations as ops

seq_len = 10
batch_size = 7

hidden_dim = 512
depth = 4

device = 'cpu'


def create_params(shape, depth):
    x = FractalTensor(TensorStorage(shape, kaleido.float32, device=device))
    x.indices = list(range(depth))
    x.initialize(torch.rand, *x.flatten_shape, device=device)
    return x


xss = FractalTensor(
    FractalTensorStorage(
        TensorStorage((1, hidden_dim), kaleido.float32, device=device)))
xss.indices = [list(range(seq_len)) for _ in range(batch_size)]
xss.initialize(torch.rand, *xss.flatten_shape, device=device)

Ws = create_params([hidden_dim, hidden_dim], depth)
Us = create_params([hidden_dim, hidden_dim], depth)
