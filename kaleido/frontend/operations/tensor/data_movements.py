from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import torch

import kaleido
from kaleido.frontend.operations.base import Access
from kaleido import Tensor

__all__ = [
    'cat',
    'permute',
    'stack',
]


class Cat(Access):
    def __call__(self, xs: Tuple, dim: int = 0):

        assert (len(xs))

        v = torch.cat([x.data for x in xs], dim=dim)
        t = Tensor(v.shape, xs[0]._type._dtype, device=xs[0].device)
        t.data = v
        return t


cat = Cat()


class Permute(Access):
    def __call__(self, x: Tensor, axes: Tuple[int]) -> Tensor:
        assert len(axes) == x.ndim
        shape = [x.shape[i] for i in axes]
        t = kaleido.Tensor(shape, x._type._dtype, device=x.device)

        t.data = x.data.permute(*axes)
        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


permute = Permute()


class Stack(Access):
    def __call__(self, xs: Tuple, dim: int = 0):

        assert (len(xs))

        v = torch.stack([x.data for x in xs], dim=dim)
        t = Tensor(v.shape, xs[0]._type._dtype, device=xs[0].device)
        t.data = v
        return t


stack = Stack()
