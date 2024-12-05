from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
import kaleido
import torch

__all__ = [
    'Elementwise',
    'Contraction',
    'Reduction',
    'Broadcast',
    'Access',
]


class BaseOp(object):
    def check_inputs(self, *inputs: Tuple):
        for x in inputs:
            if not isinstance(x, kaleido.Tensor):
                raise TypeError(f'Expected Tensor, got {type(x).__name__}.')
        d = inputs[0].device
        for x in inputs[1:]:
            if x.device != d:
                raise RuntimeError((
                    f'Expected all tensors to be on the same device, but found '
                    f'at least two devices, {d} and {x.device}!'))
        return inputs if len(inputs) > 1 else inputs[0]

    def __call__(self, *xs, **kwargs):
        for x in xs:
            if not isinstance(x.data, torch.Tensor):
                raise ValueError(
                    f'Expected torch.Tensor, but got {type(x.data)}')


class Access(object):
    pass


class Elementwise(BaseOp):
    def __call__(self, *xs, **kwargs):
        self.check_inputs(*xs)
        shape = xs[0].shape

        for x in xs[1:]:
            for s1, s2 in zip(shape, x.shape):
                if s1 != s2:
                    raise ValueError('Each input should have the same shape.')

        super(Elementwise, self).__call__(*xs)
        return kaleido.Tensor.like(xs[0])


class Contraction(BaseOp):
    def __call__(self, *xs, **kwargs):
        self.check_inputs(*xs)
        super(Contraction, self).__call__(*xs, **kwargs)

        #FIXME(ying): shape of the returned value is not equal to the shape
        # of `xs[0]`, as a result, strides is not correctly set.
        return kaleido.Tensor.like(xs[0])


class Reduction(BaseOp):
    def __call__(self, *xs, **kwargs):
        self.check_inputs(*xs)
        super(Reduction, self).__call__(*xs, **kwargs)
        #FIXME(ying): shape of the returned value is not equal to the shape
        # of `xs[0]`, as a result, strides is not correctly set.
        return kaleido.Tensor.like(xs[0])


class Broadcast(BaseOp):
    def __call__(self, *xs, **kwargs):
        """
        The first input is a small Tensor, while the second Tensor is a larger
        Tensor.
        """
        assert len(xs) == 2
        self.check_inputs(*xs)
        super(Broadcast, self).__call__(*xs, **kwargs)

        #FIXME(ying): shape of the returned value is not equal to the shape
        # of `xs[0]`, as a result, strides is not correctly set.
        return kaleido.Tensor.like(xs[0])
