"""Operations that create constant values."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
import torch

import kaleido
from kaleido.frontend.types import internal_type_to_torch_type
from kaleido.frontend.types import str_to_internal_type

__all__ = [
    'zeros',
    'ones',
    'arange',
    'rand',
    'full',
]


class ConstantOp(object):
    pass


class Zeros(ConstantOp):
    def __call__(self,
                 shape: Tuple[int] = None,
                 dtype: str = 'float',
                 device: str = None):
        if not shape:
            raise ValueError('shape is not given.')

        elem_type = str_to_internal_type(dtype)

        t = kaleido.Tensor(
            shape, dtype=elem_type, device='cpu' if device is None else device)
        t.data = torch.zeros(
            *shape,
            dtype=internal_type_to_torch_type(elem_type),
            device=device)
        return t


zeros = Zeros()


class Ones(ConstantOp):
    def __call__(self,
                 shape: Tuple[int] = None,
                 dtype: str = 'float',
                 device: str = None):
        if not shape:
            raise ValueError('shape is not given.')

        elem_type = str_to_internal_type(dtype)

        t = kaleido.Tensor(
            shape, dtype=elem_type, device='cpu' if device is None else device)
        t.data = torch.ones(
            *shape,
            dtype=internal_type_to_torch_type(elem_type),
            device=device)
        return t


ones = Ones()


class Arange(ConstantOp):
    def __call__(self, *args, dtype=kaleido.int32,
                 device='cpu') -> kaleido.Tensor:
        """Return evenly spaced values within a given interval.

        Args:
            args, Tule[int], Accept 3 inputs at most that is interpreted as:
            start, stop and step respectively.
        Returns, Tensor[int], [start, stop) with `step` as the interval.
        """

        start = 0
        stop = 0
        step = 1

        if len(args) == 1:
            stop = args[0]
        elif len(args) == 2:
            start = args[0]
            stop = args[1]
        elif len(args) == 3:
            start = args[0]
            stop = args[1]
            step = args[2]
        else:
            raise ValueError('`arange` accepts three input arguments at most.')
        if stop <= start:
            raise ValueError('stop should be larger than start.')

        l = (stop - start) // step
        t = kaleido.Tensor(
            (l, ), dtype=dtype, device='cpu' if device is None else device)
        t.data = torch.arange(
            start,
            stop,
            step,
            dtype=internal_type_to_torch_type(dtype),
            device=device)
        return t


arange = Arange()


class Random(ConstantOp):
    def __call__(self,
                 shape: Tuple[int] = None,
                 dtype=kaleido.float32,
                 device: str = None):
        if not shape:
            raise ValueError('shape is not given.')

        t = kaleido.Tensor(
            shape, dtype=dtype, device='cpu' if device is None else device)
        t.data = torch.rand(
            *shape, dtype=internal_type_to_torch_type(dtype), device=device)
        return t


rand = Random()


class Full(ConstantOp):
    """
    Creates a tensor of size size filled with fill_value. The tensor's dtype is inferred from fill_value
    """

    def __call__(self,
                 shape: Tuple[int],
                 fill_value,
                 dtype=kaleido.float32,
                 device: str = None):
        if not shape:
            raise ValueError('shape is not given.')

        t = kaleido.Tensor(
            shape, dtype=dtype, device='cpu' if device is None else device)
        t.data = torch.full(
            shape,
            fill_value=fill_value,
            dtype=internal_type_to_torch_type(dtype),
            device=device)

        return t


full = Full()
