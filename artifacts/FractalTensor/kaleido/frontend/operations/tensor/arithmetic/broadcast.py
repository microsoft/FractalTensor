from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import kaleido
from kaleido.frontend.operations.base import Broadcast

__all__ = [
    'scale',
    '_broadcast_div',
    '_broadcast_pow',
]


class Scale(Broadcast):
    """y = x * y where x is a scalar

    x is the smaller tensor while y is the larger tensor.
    """

    def __call__(self, x: kaleido.Tensor, y: kaleido.Tensor) -> kaleido.Tensor:
        t = super(Scale, self).__call__(x, y)

        t.data = x.data * y.data
        t._type._shape = t.data.shape
        t.recompute_strides()
        return t


scale = Scale()


class _BroadcastDiv(Broadcast):
    def __call__(self, x: kaleido.Tensor, y: kaleido.Tensor) -> kaleido.Tensor:
        t = super(_BroadcastDiv, self).__call__(x, y)

        t.data = x.data / y.data
        t._type._shape = t.data.shape
        t.recompute_strides()
        return t


_broadcast_div = _BroadcastDiv()


class _BroadcastPow(Broadcast):
    def __call__(self, x: kaleido.Tensor, y: kaleido.Tensor) -> kaleido.Tensor:
        t = super(_BroadcastPow, self).__call__(x, y)

        t.data = x.data**y.data
        t._type._shape = t.data.shape
        t.recompute_strides()
        return t


_broadcast_pow = _BroadcastPow()
