"""
Tensor is a mathmetical concpet. Storage is its physical representation in
computer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Generic

import torch

import kaleido
from kaleido.frontend.types import TensorStorage
from kaleido.frontend.types import Number

__all__ = [
    'Tensor',
    'Parameter',
]

Shape = TypeVar('Shape', str, bytes)
Device = TypeVar('Device', str, bytes)
DType = TypeVar('DType')
T = TypeVar('Tensor')


class Tensor(Generic[Shape, DType, Device]):
    def __init__(
            self,
            shape: Tuple[int],
            dtype: Number,
            order: str = None,  # Column-major ('col'); or Row-major ('row').
            device: str = None,
            strides: Tuple[int] = None):
        """Denoted as Tensor[T, (S...)].

        Declare a tensor variable.

        - T is elementary type including Int, Uint, Real, Boolean.
        - S is the TensorShape S =< S1,S2,...,Sn > which is a tuple of
          n non-negative integers.
          - TensorShape specifies the size of each dimension of a tensor.
          - TensorShape is a iterative collection of non-negative
            integers which are ordered and immutable.
          - Elements of TensorShape can be traversed through sequentially.
        """
        super(Tensor, self).__init__()

        if not (isinstance(shape, Tuple) or isinstance(shape, List)):
            raise TypeError('TensorShape is required.')
        for s in shape:
            if not isinstance(s, int):
                raise TypeError('TensorShape is required.')

        order = 'row' if order is None else order
        if order not in ['row', 'col']:
            raise ValueError(f'Unsuported layout: {order}.')

        device = 'cpu' if device is None else device
        if device not in ['cpu', 'cuda']:
            raise ValueError(f'Unsuported device: {device}.')

        self._type = TensorStorage(
            shape, dtype, strides=strides, order=order, device=device)

        # physical storage of the underlying data. This could be thought as a
        # pointer to physical memory.
        self._T = None

    @property
    def ndim(self) -> Tuple[int]:
        return len(self._type.shape)

    @property
    def numel(self) -> int:
        return self._type.numel

    @property
    def shape(self) -> Tuple[int]:
        return self._type.shape

    @property
    def strides(self) -> Tuple[int]:
        return self._type._strides

    @property
    def data(self):
        return self._T

    @data.setter
    def data(self, v):
        self._T = v

    @property
    def element_type(self):
        """Get element type of the Tensor."""
        return self._type._dtype

    @property
    def device(self):
        """Get device of a tensor"""
        return self._type._device

    @property
    def T(self):
        """Returns the transposition of the matrix."""
        if self.ndim > 2:
            raise TypeError('Transposition is only defined for matrix.')

        return kaleido.frontend.operations.permute(self, [1, 0])

    @staticmethod
    def from_tensor(v):
        if isinstance(v, torch.Tensor):

            def _from_torch_tensor(v):
                if v.dtype is torch.int32:
                    dtype = kaleido.int32
                elif v.dtype is torch.int64:
                    dtype = kaleido.int64
                elif v.dtype is torch.float32:
                    dtype = kaleido.float32
                else:
                    raise NotImplementedError()

                if v.device == torch.device('cpu'):
                    device = 'cpu'
                elif v.device == torch.device('cuda', index=0):
                    device = 'cuda'
                else:
                    raise NotImplementedError()

                t = Tensor(list(v.shape), dtype, device=device)
                t.data = v
                return t

            return _from_torch_tensor(v)
        else:
            raise NotImplementedError()

    @staticmethod
    def like(x):
        assert isinstance(x, kaleido.Tensor)
        return kaleido.Tensor(x.shape, x.element_type, x._type._order,
                              x.device, x.strides)

    def _scalar_to_tensor(self, x, device):
        if isinstance(x, kaleido.Tensor):
            return x
        elif isinstance(x, float):
            t = kaleido.Tensor((1, ), kaleido.float32, device=device)
            if device == 'cuda':
                t.data = torch.cuda.FloatTensor([x], device=device)
            elif device == 'cpu':
                t.data = torch.FloatTensor([x], device=device)
            else:
                raise ValueError(f'unknown device: {device}')
            return t
        elif isinstance(x, int):
            t = kaleido.Tensor((1, ), kaleido.int32, device=device)
            if device == 'cuda':
                t.data = torch.cuda.IntTensor([x], device=device)
            elif device == 'cpu':
                t.data = torch.IntTensor([x], device=device)
            else:
                raise ValueError(f'unknown device: {device}')
            return t
        else:
            raise TypeError()

    def size(self, n):
        if n >= self.ndim:
            raise ValueError("`n` should be less than tensor's rank.")
        return self.shape[n]

    def __str__(self):
        return str(self._type)

    def is_equal_shape(self, s):
        for s1, s2 in zip(s.shape, self.shape):
            if s1 != s2:
                return False
        return True

    def recompute_strides(self):
        self._type.recompute_strides()

    __repr__ = __str__

    def __radd__(self, y: T) -> T:
        """self + y"""
        y = self._scalar_to_tensor(y, self.device)
        return kaleido.frontend.operations.add(self, y)

    __add__ = __radd__

    def __rsub__(self, y: T):
        return kaleido.frontend.operations.sub(self, y)

    __sub__ = __rsub__

    def __rmul__(self, y: T):
        """y * self"""
        y = self._scalar_to_tensor(y, self.device)
        if self.is_equal_shape(y):
            return kaleido.frontend.operations.multiply(self, y)
        else:
            return kaleido.frontend.operations.scale(self, y)

    __mul__ = __rmul__

    def __matmul__(self, y: T):
        """self @ y"""
        return kaleido.frontend.operations.mm(self, y)

    def __rmatmul__(self, y: T):
        raise NotImplementedError()

    def __pow__(self, y: T):
        y = self._scalar_to_tensor(y, self.device)
        return kaleido.frontend.operations.pow(self, y)

    def __ipow__(self, y: T):
        raise NotImplementedError()

    def __rpow__(self, y: T):
        if not isinstance(y, int):
            raise TypeError(f'Unsupported type {type(y)}.')
        y = self._scalar_to_tensor(y, self.device)

        if self.numel == y.numel:
            return kaleido.frontend.operations.pow(y, self.data.item())
        else:
            return kaleido.frontend.operations._broadcast_pow(y, self)

    def __rtruediv__(self, y: T):
        y = self._scalar_to_tensor(y, self.device)

        if y.numel == self.numel:
            return kaleido.frontend.operations.div(y, self)
        else:
            return kaleido.frontend.operations._broadcast_div(y, self)

    def __truediv__(self, y: T):
        y = self._scalar_to_tensor(y, self.device)

        if y.numel == self.numel:
            return kaleido.frontend.operations.div(self, y)
        else:
            return kaleido.frontend.operations._broadcast_div(self, y)

    def __mod__(self, y: T):
        return self.data % y

    def __eq__(self, y: T):
        if isinstance(self.element_type, kaleido.frontend.types.Int):
            return int(self.data) == y
        elif isinstance(self.element_type, kaleido.frontend.types.Real):
            return float(self.data) == y
        else:
            raise NotImplementedError()

    def __ne__(self, y: T):
        if isinstance(self.element_type, kaleido.frontend.types.Int):
            return int(self.data) != y
        elif isinstance(self.element_type, kaleido.frontend.types.Real):
            return float(self.data) != y
        else:
            raise NotImplementedError()

    def __gt__(self, y: T):
        if isinstance(self.element_type, kaleido.frontend.types.Int):
            return int(self.data) > y
        elif isinstance(self.element_type, kaleido.frontend.types.Real):
            return float(self.data) > y
        else:
            raise NotImplementedError()

    def __ge__(self, y: T):
        if isinstance(self.element_type, kaleido.frontend.types.Int):
            return int(self.data) >= y
        elif isinstance(self.element_type, kaleido.frontend.types.Real):
            return float(self.data) >= y
        else:
            raise NotImplementedError()

    def __lt__(self, y: T):
        if isinstance(self.element_type, kaleido.frontend.types.Int):
            return int(self.data) < y
        elif isinstance(self.element_type, kaleido.frontend.types.Real):
            return float(self.data) < y
        else:
            raise NotImplementedError()

    def __le__(self, y: T):
        if isinstance(self.element_type, kaleido.frontend.types.Int):
            return int(self.data) <= y
        elif isinstance(self.element_type, kaleido.frontend.types.Real):
            return float(self.data) <= y
        else:
            raise NotImplementedError()

    def initialize(self, initializer: callable, *args, **kwargs):
        self._T = initializer(*args, **kwargs)

    def view(self, shape):
        return kaleido.operations.reshape(self, shape)

    def is_equal_type(self, y):
        assert isinstance(y, Tensor)
        return self._type.is_equal_type(y._type)


class Parameter(Tensor, Generic[Shape, DType, Device]):
    """
    Learnable parameter is a mutable tensors and has a dual tensor to store
    gradients, or momentum if needed by the algorithm. More importantly in the
    program analysis, a Parameter is a definition of a value.
    """

    def __init__(
            self,
            shape: Tuple[int],
            dtype,
            order: str = None,  # Column-major ('col'); or Row-major ('row').
            device: str = None,
            strides: Tuple[int] = None):
        super(Parameter, self).__init__(shape, dtype, order, device)

    def update(self, func: callable):
        pass
