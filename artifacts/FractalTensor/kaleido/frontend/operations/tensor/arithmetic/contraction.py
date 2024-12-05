from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
import torch

import kaleido
from kaleido.frontend.operations.base import Contraction

__all__ = [
    'mm',
    'outer',
]


class MatMul(Contraction):
    """ (tensor contraction: reduce + map) matrix multiplication

    y = a \otimes b
    """

    def __call__(self, x: kaleido.Tensor, y: kaleido.Tensor) -> kaleido.Tensor:
        t = super(MatMul, self).__call__(x, y)

        t.data = x.data @ y.data
        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


mm = MatMul()


class Outer(Contraction):
    """ Outer product of two vector."""

    def __call__(self, x: kaleido.Tensor, y: kaleido.Tensor) -> kaleido.Tensor:
        t = super(Outer, self).__call__(x, y)

        t.data = torch.outer(x.data, y.data)
        t._type._shape = list(t.data.shape)

        t.recompute_strides()
        return t


outer = Outer()
