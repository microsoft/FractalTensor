# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import random
from typing import NamedTuple, Tuple

import torch

import kaleido
from kaleido import FractalTensor, FractalTensorStorage, Tensor, TensorStorage
from kaleido import operations as ops


def gen_image_batch(tensor_shape: Tuple[int],
                    batch_size: int,
                    device='cpu') -> FractalTensor[FractalTensor[Tensor]]:
    """Returns a batch of image in format of NCHW."""
    x = FractalTensor(
        TensorStorage(tensor_shape, kaleido.float32, device=device))
    x.indices = list(range(batch_size))
    x.initialize(torch.rand, *x.flatten_shape, device=device)
    return x
