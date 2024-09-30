from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import kaleido
from kaleido.frontend.operations.base import Elementwise

__all__ = [
    'add',
    'sub',
    'multiply',
    'div',
    'exp',
    'log',
    'pow',
    'sin',
    'cos',
    'tanh',
    'sqrt',
    'abs',
    'sigmoid',
    'cross_entropy',
    'dropout',
    'relu',
    'maximum',
]


def _equal_shape(s1, s2):
    for x, y in zip(s1, s2):
        if x != y:
            return False
    return True


class Add(Elementwise):
    """Element-wise add. y = a + b"""

    def __call__(self, x, y):
        if not _equal_shape(x.shape, y.shape):
            # FIXME(ying): hot fix for broadcast add, and x is the big Tensor.
            t = kaleido.Tensor.like(x)
            t.data = x.data + y.data
            t._type._shape = list(t.data.shape)
            t.recompute_strides()
            return t

        t = super(Add, self).__call__(x, y)

        t.data = x.data + y.data
        return t


add = Add()


class Sub(Elementwise):
    """element-wise substraction: y = a - b."""

    def __call__(self, x, y):
        if not _equal_shape(x.shape, y.shape):
            # FIXME(ying): hot fix for broadcast sub, and x is the big Tensor.
            t = kaleido.Tensor.like(x)
            t.data = x.data - y.data

            t._type._shape = list(t.data.shape)
            t.recompute_strides()
            return t

        t = super(Sub, self).__call__(x, y)

        t.data = x.data - y.data
        return t


sub = Sub()


class Multiply(Elementwise):
    """y = a * b"""

    def __call__(self, x, y):
        t = super(Multiply, self).__call__(x, y)

        t.data = x.data * y.data

        t._type._shape = list(t.data.shape)
        t.recompute_strides()
        return t


multiply = Multiply()


class Div(Elementwise):
    """Elementwise div.

    y = a / b
    """

    def __call__(self, x, y):
        t = super(Div, self).__call__(x, y)

        t.data = x.data / y.data
        return t


div = Div()


class Exp(Elementwise):
    """exp"""

    def __call__(self, x):
        t = super(Exp, self).__call__(x)

        t.data = torch.exp(x.data)
        return t


exp = Exp()


class Log(Elementwise):
    """log(x)"""

    def __call__(self, x):
        t = super(Log, self).__call__(x)

        t.data = torch.log(x.data)
        return t


log = Log()


class Pow(Elementwise):
    """Elementwise pow.

    y = a ** b
    """

    def __call__(self, x, y: float):
        t = super(Pow, self).__call__(x)

        t.data = torch.pow(x.data, y)
        return t


pow = Pow()


class Sin(Elementwise):
    """sin(x)"""

    def __call__(self, x):
        t = super(Sin, self).__call__(x)

        t.data = torch.sin(x.data)
        return t


sin = Sin()


class Cos(Elementwise):
    """sin(x)"""

    def __call__(self, x):
        t = super(Cos, self).__call__(x)

        t.data = torch.cos(x.data)
        return t


cos = Cos()


class Tanh(Elementwise):
    """tanh(x)"""

    def __call__(self, x: kaleido.Tensor) -> kaleido.Tensor:
        t = super(Tanh, self).__call__(x)

        t.data = torch.tanh(x.data)
        return t


tanh = Tanh()


class Sqrt(Elementwise):
    """sqrt(x)"""

    def __call__(self, x):
        t = super(Sqrt, self).__call__(x)

        t.data = torch.sqrt(x.data)
        return t


sqrt = Sqrt()


class Abs(Elementwise):
    """elementwise asb(x)"""

    def __call__(self, x):
        t = super(Abs, self).__call__(x)

        t.data = torch.abs(x.data)
        return t


abs = Abs()


class Dropout(Elementwise):
    """dropout(x)"""

    def __call__(self, x, drop_rate=0.5):
        t = super(Dropout, self).__call__(x)

        t.data = torch.nn.functional.dropout(x.data, p=drop_rate)
        return x


dropout = Dropout()


class Relu(Elementwise):
    """max(0, x)"""

    def __call__(self, x):
        t = super(Relu, self).__call__(x)

        t.data = torch.nn.functional.relu(x.data)
        return x


relu = Relu()


class Sigmoid(Elementwise):
    """sigmoid

    y = 1 + \frac{1}{\text{exp}(-x)}
    """

    def __call__(self, x):
        t = super(Sigmoid, self).__call__(x)

        t.data = torch.sigmoid(x.data)
        return t


sigmoid = Sigmoid()


class CrossEntropy(Elementwise):
    """p*log(p)"""

    def __call__(self, x: kaleido.Tensor, y: kaleido.Tensor) -> kaleido.Tensor:
        t = super(CrossEntropy, self).__call__(x, y)

        t.data = torch.nn.functional.cross_entropy(x.data, y.data)
        shape = list(t.data.shape)
        t._type._shape = shape if shape else [1]
        return t


cross_entropy = CrossEntropy()


class Maximum(Elementwise):
    """Element-wise max. y = max(a, b)"""

    def __call__(self, x, y):
        t = super(Maximum, self).__call__(x, y)
        t.data = torch.maximum(x.data, y.data)
        return t


maximum = Maximum()
