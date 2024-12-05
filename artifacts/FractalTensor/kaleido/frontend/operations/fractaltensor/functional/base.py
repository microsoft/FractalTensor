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


class Functional(object):
    def _check_iterative(self, xs):
        """check whether xs is iterative."""

        if isinstance(xs, Iterative):
            return
        elif isinstance(xs, FractalTensor):
            return
        else:
            raise TypeError(
                'Input should either a FractalTensor or a Iterative.')

    def _check_types(self, x):
        if isinstance(x, Tensor):
            return
        elif isinstance(x, FractalTensor):
            return
        elif isinstance(x, Iterative):
            return
        else:
            raise TypeError(('Expected Tensor, FractalTensor or Iterative, '
                             f'but {type(x)} is given.'))

    def _check_tensor_and_tensorarrray(self, x):
        if isinstance(x, Tensor):
            return
        elif isinstance(x, FractalTensor):
            return
        elif isinstance(x,
                        int):  # hotfix for enumerate's first returned value.
            return
        else:
            raise TypeError(('Expected Tensor, ir FractalTensor, '
                             f'but {type(x)} is given.'))

    def _create_fractaltensor(self, values_to_pack: List,
                              dtype) -> FractalTensor:
        if dtype is Tensor:
            return FractalTensor.from_tensors(*values_to_pack)
        elif dtype is FractalTensor:
            return FractalTensor.from_fractaltensors(*values_to_pack)
        elif dtype is int:
            return values_to_pack
        else:
            raise TypeError(f'Unsupported type {dtype}.__name__.')

    def _pack_outputs(self, results):
        if isinstance(results[0], Tensor):
            return FractalTensor.from_tensors(*results)
        elif isinstance(results[0], FractalTensor):
            return FractalTensor.from_fractaltensors(*results)
        elif isinstance(results[0], Tuple) or isinstance(results[0], List):
            # the lambda function returns more than 1 returned value with a type
            # of either Tensor or FractalTensor
            rv_num = len(results[0])
            for x in results[1:]:
                if len(x) != rv_num:
                    raise RuntimeError(
                        'inconsistent behavior of calling lambda function.')

            for x in results[0]:
                self._check_tensor_and_tensorarrray(x)

            rvs = []
            for i in range(rv_num):
                dtype = type(results[0][i])
                v = self._create_fractaltensor(
                    [result[i] for result in results], dtype)
                rvs.append(v)
            return rvs
        elif isinstance(results[0], Iterative):
            rv_num = len(results[0][0])
            for x in results[1:]:
                if len(x[0]) != rv_num:
                    raise RuntimeError(
                        'inconsistent behavior of calling lambda function.')

            for x in results[0][0]:
                self._check_tensor_and_tensorarrray(x)

            final_results = []
            dtype = type(results[0][0][0])
            for i in range(rv_num):
                tmp_rv = []
                for result in results:
                    tmp_rv.append(
                        self._create_fractaltensor([x[i] for x in result],
                                                   dtype))
                final_results.append(
                    FractalTensor.from_fractaltensors(*tmp_rv))
            return final_results
        else:
            raise ValueError("Unsupported type of returned value.")
