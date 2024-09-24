"""Looping patterns with explicit access patterns and data dependence information.
Follows iterative protocol."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
from typing import TypeVar

import kaleido
from kaleido import Tensor
from kaleido import FractalTensor
from kaleido import Iterative
from kaleido.frontend.operations.fractaltensor.functional.base import Functional

__all__ = [
    'fold',
    'scan',
    'reduce',
]

T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')


class Scan(Functional):
    def __call__(self,
                 func: Callable[[T1, T2], T1],
                 input: Union[FractalTensor, Iterative],
                 initializer: T1 = None,
                 **kwargs) -> Union[FractalTensor, Tuple[FractalTensor]]:
        """
        Args:
            func, Callable, func is a binary operator. T1 rv = func(T1 a, T2 b)
                  where rv's and a's type should be the same as `initializer` if
                  `initializer` is not None.
            input, Union[FractalTensor, Tuple].
            initializer, T1. Initial value to scan.
                         If initializer is not None:
                             func(func(func(initializer, input[0]), input[1]), ...)
                         else:
                            func(func(func(input[0], input[1]), input[2]), ...)
        Returns:
            Union[FractalTensor, Tuple[FractalTensor]]
        """

        self._check_iterative(input)

        state = initializer if initializer is not None else input[0]
        start = 0 if initializer is not None else 1

        results = [state] if initializer is None else []
        for i in range(start, len(input), 1):
            state = func(state, input[i], **kwargs)
            results.append(state)

        return self._pack_outputs(results)


scan = Scan()


class Fold(Scan):
    def __call__(self,
                 func: Callable[[T1, T2], T1],
                 input: Union[FractalTensor, Iterative],
                 initializer: T1 = None,
                 **kwargs) -> T1:
        """Return the last element of scan."""

        outs = super(Fold, self).__call__(
            func, input, initializer=initializer, **kwargs)

        if isinstance(outs, FractalTensor):
            if not outs.indices:
                raise RuntimeError('FractalTensor indices is missing.')
            return outs.last()

        elif isinstance(outs, Tuple) or isinstance(outs, List):
            if len(outs) == 1:
                return outs[0].last()

            for elem in outs:
                if not isinstance(elem, FractalTensor):
                    raise TypeError(
                        '`outs` should be a tuple of FractalTensor.')
            return [out.last() for out in outs]
        else:
            raise TypeError(
                '`outs` should be a FractalTensor, or Tuple of FractalTensor.')


fold = Fold()


class Reduction(Fold):
    def __call__(self,
                 func: Callable[[T1, T2], T1],
                 input: Union[FractalTensor, Iterative],
                 initializer: T1 = None,
                 **kwargs) -> T1:
        return super(Reduction, self).__call__(
            func, input, initializer=initializer, **kwargs)


reduce = Reduction()
