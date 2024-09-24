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
    'map',
    'filter',
    'dilated_map',
]

T = TypeVar('T')
T1 = TypeVar('T1')
TA = TypeVar('TA')


class Map(Functional):
    def __call__(self, func: Callable[[T1], T],
                 xs: Union[Iterative, FractalTensor],
                 **kwargs) -> Union[Tuple[FractalTensor], FractalTensor]:
        """Apply `func` to each element in the FractalTensor or the Iterative.

        Args:
            func, callable,
        """

        self._check_iterative(xs)

        results = []
        for x in xs:
            results.append(func(x, **kwargs))

        rv = self._pack_outputs(results)
        return rv


map = Map()


class Filter(Functional):
    def __call__(self, condition: Callable[[T1], bool],
                 xs: Union[Iterative, FractalTensor],
                 **kwargs) -> Union[Tuple[FractalTensor], FractalTensor]:
        """Apply `func` to each element in the FractalTensor or the Iterative.

        Args:
            condition, Callable, a user function that returns a bool,

        Returns:
            a FractalTensor constructed from applying `condition` to
            each element of xs. The returned FractalTensor holds elements in
            `xs` that fulfill the condition given by `predicate`.
        """

        self._check_iterative(xs)

        results = []
        for x in xs:
            if condition(x):
                results.append(x)

        if len(results):
            return self._pack_outputs(results)
        return None


filter = Filter()


class DilatedMap(Functional):
    def _dilate(self, x: FractalTensor[T],
                dilation: int) -> FractalTensor[FractalTensor[T]]:
        """Dilate input x adaptive to its length.

        Returns:
            FractalTensor[FractalTensor[T]] with an increased depth by 1.

        Example 1:
            Input:
                x = [0, 1, 2, 3, 4, 5, 6, 7, 8], dilation = 2:
            Returns:
                [
                  [0, 2, 4, 6, 8],
                  [1, 3, 5, 7],
                ]

        Example 2:
            Input:
                x = [0, 1, 2, 3, 4, 5, 6, 7, 8], dilation = 3:
            Returns:
                [
                  [0, 3, 6],
                  [1, 4, 7],
                  [2, 5, 8]
                ]

        Example 3:
            Input:
                x = [0, 1, 2, 3, 4, 5, 6, 7, 8], dilation = 5:
            Returns:
                [
                  [0, 5],
                  [1, 6],
                  [2, 7],
                  [3, 8],
                  [4],
                ]
        """
        if not isinstance(x, FractalTensor):
            raise TypeError('x should be a FractalTensor.')

        if dilation <= 0:
            raise ValueError('dilation should be greater than 0.')

        if len(x) <= dilation or dilation == 1:
            return FractalTensor.from_fractaltensors(x)

        slided = []
        for i in range(dilation):
            indices = list(range(i, len(x), dilation))
            slided.append(
                FractalTensor.from_tensors(*[x[ids] for ids in indices]))
        v = FractalTensor.from_fractaltensors(*slided)
        return v

    def _revert_dilate(self, x: FractalTensor[T], dilation: int) -> T:
        # FIXME(ying): rethink of dilated_map, current implementation is a hot fix.
        if dilation == 1:
            assert len(x) == 1
            return x[0]
        assert dilation == len(x)

        items = []
        max_len = len(x[0])
        for i in range(max_len):
            for j in range(dilation):
                if i == len(x[j]):
                    continue
                items.append(x[j][i])
        assert x.numel == len(items)
        if x.depth == 2:
            return FractalTensor.from_tensors(*items)
        else:
            return FractalTensor.from_fractaltensors(*items)

    def __call__(self,
                 func: Callable,
                 xs: Union[Iterative, FractalTensor],
                 dilation: int = 1,
                 **kwargs):
        self._check_iterative(xs)
        dilation_ = dilation
        if isinstance(dilation, Tensor):
            dilation_ = int(dilation.data)

        rvs = kaleido.frontend.operations.map(lambda vs: func(vs, **kwargs),
                                              self._dilate(xs, dilation_))

        results = []
        for rv in rvs:
            v = self._revert_dilate(rv, dilation_)
            results.append(v)
        return results[0] if len(results) == 1 else results


dilated_map = DilatedMap()
