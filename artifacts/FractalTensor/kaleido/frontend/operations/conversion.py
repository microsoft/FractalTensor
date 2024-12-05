from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Union
from typing import Tuple
from typing import List
from typing import TypeVar

import torch

T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')

import copy
import functools
import operator

import kaleido
from kaleido.frontend.operations.base import Access
from kaleido import Tensor
from kaleido import TensorStorage
from kaleido import FractalTensor
from kaleido import FractalTensorStorage

__all__ = [
    'stack',
    'unbind',
    'flatten',
    'slices',
    'repeat',
    'split',
]


class Flatten(Access):
    def __call__(self, x: FractalTensor, dim: int = 0) -> Tensor:
        """Convert a nested FractalTensor variable into a Tensor variable.

        Elements in a nested FractalTensor are stored contingously according to
        the positive lexicographical order of their indices in the FractalTensor.

        Suppose `x:FractalTensor[Tensor(shape_x, dtype_x)]` is a nested FractalTensor
        varialbe. The returned value 'y' is with a shape of (numel(x)) + shape_x.

        Examples:
                x: FractalTensor[Tensor[(13, 27), float32]] =
            =   [
                    [a, b, c],
                    [d, e]
                ]

                y: Tensor[(numel(x), 13, 27), float32] = tensorize(x)
            = [a, b, c, d, e]

            where a, b, c, d, e are tensors with a shape of (13, 27)

        Returns:
            Tensor.
        """

        if not isinstance(x, FractalTensor):
            raise TypeError('input x should be a FractalTensor variable.')

        new_shape = list(x.element_shape)

        if functools.reduce(operator.mul, new_shape, 1) == 1:
            new_shape = [1]

        assert len(x.element_type.shape) > dim
        new_shape[dim] = x.numel * new_shape[dim]

        t = Tensor(tuple(new_shape), x.element_type._dtype, device=x.device)
        t.data = x.data.view(new_shape)
        t.recompute_strides()
        return t


flatten = Flatten()


class Stack(Access):
    def __call__(self, xs: List[Tensor], dim: int = 0):
        assert (len(xs))

        v = torch.stack([x.data for x in xs], dim=dim)
        t = Tensor(v.shape, xs[0]._type._dtype, device=xs[0].device)
        t.data = v
        return t


stack = Stack()


class Slices(Access):
    def __call__(self, x: Tensor, dim: int) -> FractalTensor:
        """Return slices of a tensor along the given dim.

        Input `x` and returned value share the same underlying memory.

        Returns:
            a 1-depth FractalTensor.
        """
        if not isinstance(x, Tensor):
            raise TypeError('input x should be a tensor.')

        length = x.shape[dim]
        new_shape = list(x.shape)
        del new_shape[dim]

        if len(new_shape) == 0:
            new_shape = [1]
        elif len(new_shape) == 1:
            new_shape = [1] + new_shape  # vector is interperted as row vector.

        if dim != 0:
            axes = list(range(x.ndim))
            axes[0] = axes[dim]
            axes[dim] = 0

            x = kaleido.frontend.operations.permute(x, axes)

        ta = FractalTensor(
            TensorStorage(new_shape, x._type._dtype, device=x.device))
        ta.indices = list(range(length))
        ta.data = x.data.reshape(-1)
        return ta


slices = Slices()


class Unbind(Access):
    def __call__(self, xs: FractalTensor):
        return [x for x in xs]


unbind = Unbind()


class Repeat(Access):
    def __call__(self, a: Union[Tensor, FractalTensor[T]], repeats: int
                 ) -> Union[FractalTensor[T], FractalTensor[FractalTensor[T]]]:
        """Repeat a FractalTensor.

        Args:
            a, FractalTensor,
            repeats, int, how many times to repeat.

        Returns:
            a FractalTensor with an increased depth by 1.
        """
        if isinstance(a, FractalTensor):
            ta_type = FractalTensorStorage(a.element_type)
            t = FractalTensor(ta_type)

            if a.indices is not None:
                # runtime hit this branch.
                indices = [copy.deepcopy(a.indices) for _ in range(repeats)]
                t.indices = indices
                t.data = a.data.repeat(repeats)
            return t
        elif isinstance(a, Tensor):
            t = FractalTensor(a._type)

            if a.data is not None:
                t.indices = list(range(repeats))
                t.data = a.data.repeat_interleave(repeats)

            return t
        else:
            raise TypeError('input a should be a Tensor or a FractalTensor.')


repeat = Repeat()


class Split(Access):
    def __call__(self,
                 a: Union[Tensor, FractalTensor[T]],
                 num: int,
                 dim: int = -1) -> FractalTensor:
        """Split a FractalTensor into n parts.

        Args:
            a, Union[Tensor, FractalTensor],
            num, int, number of parts to split.

        Returns:
            a FractalTensor with an increased depth by 1.
        """
        if isinstance(a, FractalTensor):
            ta_type = FractalTensorStorage(a.element_type)
            t = FractalTensor(ta_type)

            l = a.length // num
            splits = []
            for i in range(num - 1):
                v = a[i * l:(i + 1) * l]
                splits.append(v)
            splits.append(a[(num - 1) * l:])
            return splits
        elif isinstance(a, Tensor):
            if dim == -1:
                raise ValueError(('When input a is a Tensor, '
                                  'dims should be specified.'))
            new_shape = list(a.shape)
            split_size = new_shape[dim] // num
            new_shape[dim] = split_size
            ft = FractalTensor(
                TensorStorage(new_shape, a._type._dtype, device=a.device))
            ft.indices = list(range(num))
            # FIXME(ying): not the correct layout.
            ft.data = a.data
            return ft
        else:
            raise TypeError(f'Except FractalTensor, got {type(a).__name__}.')


split = Split()
