from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = [
    'reshape',
    'squeeze',
    'unsqueeze',
]

import torch

from typing import Union
from typing import List

import kaleido
from kaleido.frontend.operations.base import BaseOp


class Reshape(BaseOp):
    def __call__(self, x: kaleido.Tensor, shape: Union[List[int], int]):
        if not isinstance(shape, List):
            if isinstance(shape, int):
                shape = [shape]
            else:
                raise ValueError('shape should be list of integers.')

        super(Reshape, self).__call__(x)

        t = kaleido.Tensor(shape, x._type._dtype, device=x.device)
        t.data = torch.reshape(x.data, shape)
        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


reshape = Reshape()


class Squeeze(BaseOp):
    def __call__(self, x: kaleido.Tensor, dim: int = None):
        super(Squeeze, self).__call__(x)

        t = kaleido.Tensor([0], x._type._dtype, device=x.device)
        t.data = x.data.squeeze(dim) if dim else x.data.squeeze()
        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


squeeze = Squeeze()


class Unsqueeze(BaseOp):
    def __call__(self, x: kaleido.Tensor, dim: int):
        super(Unsqueeze, self).__call__(x)

        t = kaleido.Tensor([0], x._type._dtype, device=x.device)
        t.data = x.data.unsqueeze(dim)
        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


unsqueeze = Unsqueeze()
