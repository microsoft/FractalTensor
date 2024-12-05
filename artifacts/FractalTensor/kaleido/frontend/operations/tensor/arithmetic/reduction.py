from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kaleido
import pdb
from kaleido.frontend.operations.base import Reduction

__all__ = [
    'max',
    'sum',
    'mean',
    'dot',
    'softmax',
    'layer_norm',
]

import torch


class Softmax(Reduction):
    """softmax(0, x)"""

    def __call__(self, x, dim: int = 0):
        t = super(Softmax, self).__call__(x, dim=dim)

        t.data = torch.nn.functional.softmax(x.data, dim=dim)

        #TODO(ying) Fix shape inference and strides.
        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


softmax = Softmax()


class LayerNorm(Reduction):
    def __call__(self, x, w, b, eps=1e-6):
        t = super(LayerNorm, self).__call__(x)

        t.data = torch.nn.functional.layer_norm(
            x.data, x.shape, weight=w.data, bias=b.data, eps=eps)
        #TODO(ying) Fix shape inference.
        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


layer_norm = LayerNorm()


class Max(Reduction):
    """ reduced max: y = max(x) """

    def __call__(self, x, dim=0, keepdim=False):
        t = super(Max, self).__call__(x, dim=dim, keepdim=keepdim)

        t.data = torch.max(x.data, dim=dim, keepdim=keepdim).values
        #TODO(ying) Fix shape inference.
        t._type._shape = list(t.data.shape)
        if not t._type._shape:
            t._type._shape = [1]  # hot fix for scalar
        t.recompute_strides()
        return t


max = Max()


class Min(Reduction):
    """ reduced min: y = max(x) """

    def __call__(self, x, dim=0, keepdim=False):
        t = super(Min, self).__call__(x, dim=dim, keepdim=keepdim)

        t.data = torch.min(x.data, dim=dim, keepdim=keepdim)
        #TODO(ying) Fix shape inference.
        t._type._shape = list(t.data.values.shape)
        t.recompute_strides()
        return t


min = Min()


class Sum(Reduction):
    """ reduced sum: y = sum(x) """

    def __call__(self, x, **kwargs):
        t = super(Sum, self).__call__(x, **kwargs)

        t.data = torch.sum(x.data, **kwargs)
        #TODO(ying) Fix shape inference.
        t._type._shape = list(t.data.shape)
        if len(t._type._shape) == 0:
            t._type._shape = [1]
        t.recompute_strides()
        return t


sum = Sum()


class Mean(Reduction):
    """ reduced mean: y = sum(x) """

    def __call__(self, x, dim=0, keepdim=False):
        t = super(Mean, self).__call__(x, dim=dim, keepdim=keepdim)

        t.data = torch.mean(x.data, dim=dim, keepdim=keepdim)
        #TODO(ying) Fix shape inference.
        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


mean = Mean()


class Dot(Reduction):
    """y = a * b"""

    def __call__(self, x, y, keepdim=False):
        t = super(Dot, self).__call__(x, y, keepdim=False)

        for s1, s2 in zip(x.shape, y.shape):
            if s1 != s2:
                raise RuntimeError(('x and y should have the same shape, got '
                                    f'x:{x.shape} and y:{y.shape}'))

        t.data = torch.dot(x.data.reshape(-1), y.data.reshape(-1))
        #TODO(ying) Fix shape inference for scalar.
        shape = list(t.data.shape)
        if len(shape) == 0:
            shape = [1]

        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


dot = Dot()
