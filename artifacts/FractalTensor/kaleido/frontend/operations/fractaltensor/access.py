from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union
from typing import Tuple
from typing import List
from typing import TypeVar

import torch
import copy

import kaleido
from kaleido.frontend.operations.base import Access
from kaleido import Tensor
from kaleido import TensorStorage
from kaleido import FractalTensor
from kaleido import FractalTensorStorage
from kaleido import Iterative

__all__ = [
    'index',
    'zip',
    'enumerate',
    'slide',
    'window',
    'shifted_slide',
    'shifted_window',
    'product',
    'last',
    'join',
]

T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')


class Index(Access):
    def __call__(self, a: Union[FractalTensor[T], Iterative[T]],
                 ids: Union[int, Tensor['(1,)', int, 'cpu']]) -> T:
        def _to_int(ids):
            if isinstance(ids, int):
                return ids

            if (isinstance(ids, Tensor)
                    and (ids._type._dtype is kaleido.int32
                         or ids._type._dtype is kaleido.int64)
                    and ids.numel == 1):
                return ids.data[0]
            else:
                raise ValueError('`ids` should be an integer.')

        index = _to_int(ids)

        if isinstance(a, FractalTensor) or isinstance(a, Iterative):
            return a[index]
        else:
            raise TypeError('Expected a FractalTensor or an Iterative.')


index = Index()


class Zip(Access):
    def __call__(self, *inputs: Union[FractalTensor, Iterative]) -> Iterative:
        """Make an iterator that aggregates elements from each of the iterables.

        Example 1:
            zip([a, b, c], [1, 2, 3], [A, B, C]) ->
               [[a, 1, A], [b, 2, 3], [c, 3, C]]

        Example 2:
            a = [                    # depth 2
                    [a, b, c],       # depth 1
                    [d, e]
                ]
            b = [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8]
                ]
            c = [
                    [A, B, C, D],
                    [E]
                ]

            d = zip(a, b, c)
              = [                      # depth 3
                    [                  # depth 2
                        [a, b, c],     # depth 1
                        [1, 2, 3, 4],
                        [A, B, C, D]
                    ],
                    [
                        [d, e],
                            [5, 6, 7, 8],
                            [E]
                        ]
                    ]

        Example 3:
            a = [                    # depth 3
                    [                # depth 2
                        [a, b]       # depth 1
                        [c, d, e]
                    ]
                    [
                        [f]
                        [g, h, i, j]
                        [k, l]
                    ]
                ]

            b = [
                    [
                        [1, 2]
                        [3, 4, 5, 6]
                        [7, 8, 9, 10]
                    ]
                    [
                        [11, 12]
                    ]
                ]

            c = zip(a, b)
              = [                          # depth 4
                    [                      # depth 3
                        [                  # depth 2
                            [a, b]         # depth 1
                            [c, d, e]
                        ]
                        [
                            [1, 2]
                            [3, 4, 5, 6]
                            [7, 8, 9, 10]
                        ]
                    ]
                    [
                        [
                            [f]
                            [g, h, i, j]
                            [k, l]
                        ]
                        [
                            [11, 12]
                        ]
                    ]
                ]

        """

        if len(inputs) == 1:
            if isinstance(inputs[0], Tensor) or isinstance(
                    inputs[0], FractalTensor):
                return inputs[0]
            else:
                raise TypeError(
                    f'Unsupported type of input: {type(inputs[0])}')

        for x in inputs:
            if not (isinstance(x, FractalTensor) or isinstance(x, Iterative)):
                raise TypeError(
                    '`input` to zip should be either FractalTensor or Iterative.'
                )

        num_input = len(inputs)
        length = len(inputs[0])

        for x in inputs[1:]:
            if len(x) != length:
                raise RuntimeError(
                    "Multiple inptus to zip should have the same length.")

        rv = []
        for i in range(length):
            item = [inputs[j][i] for j in range(num_input)]
            rv.append(item)
        return Iterative.make_iterative(*rv)


zip = Zip()


class Enumerate(Access):
    def __call__(self, *inputs: FractalTensor) -> Tuple[int, FractalTensor]:
        assert len(inputs) >= 1
        if not isinstance(inputs[0], FractalTensor):
            raise TypeError(('Expected FractalTensor, '
                             f'got {type(inputs[0]).__name__}'))

        length = inputs[0].length
        for x in inputs[1:]:
            if x.length != length:
                raise ValueError(
                    'Inputs to enumerate should have a same length.')
            if not isinstance(x, FractalTensor):
                raise TypeError(('Expected FractalTensor, '
                                 f'got {type(x).__name__}.'))

        ids = FractalTensor.from_pylist(list(range(length)))
        return kaleido.frontend.operations.zip(ids, *inputs)


enumerate = Enumerate()


class Product(Access):
    def __call__(self, xs: FractalTensor[T1],
                 ys: FractalTensor[T2]) -> FractalTensor:
        """cartesian product of two FractalTensors, equivalent to a nested for-loop.

        Imagine that the first argument x is arranged along the horizontal dim,
        and the second argument y is arranged along the vertical dim.
        The cartesian product expands x and y respectively into FractalTensors with
        an increased depth by 1. For each returned result, the length of the
        horizontal dim is defined by the length of x, and the length of the
        vertical dim is defined by the length of y.


        Args:
            xs, FractalTensor[T1],
            ys, FractalTensor[T2],

            `xs` and `ys` are nestable FractalTensors. There is no constraints
            on the depth of `xs` and `ys`.

        Returns:
            Tuple[FractalTensor],
        """

        if not (isinstance(xs, FractalTensor)
                and isinstance(ys, FractalTensor)):
            raise TypeError('Type of both inputs should be FractalTensor.')

        xss = kaleido.frontend.operations.repeat(xs, len(ys))

        ys_tmp = []
        for i in range(len(ys)):
            ys_tmp.append(kaleido.frontend.operations.repeat(ys[i], len(xs)))
        yss = FractalTensor.from_fractaltensors(*ys_tmp)

        return xss, yss


product = Product()


class Last(Access):
    def __call__(self, x: FractalTensor):
        return x[-1]


last = Last()


class Join(Access):
    def __call__(self, x: FractalTensor, y: FractalTensor):
        assert isinstance(x, FractalTensor) and isinstance(y, FractalTensor)
        assert x._type.is_equal_type(y._type)

        rv_type = copy.deepcopy(x.element_type)
        for i in range(x.depth - 1):
            rv_type = FractalTensorStorage(x.element_type)
        joined = FractalTensor(rv_type)
        joined.indices = (x.indices + y.indices
                          if x.depth > 1 else list(range(len(x) + len(y))))

        if x.data is not None and y.data is not None:
            # FIXME(ying): hardcoded to use torch tensor operatons.
            joined.data = torch.cat((x.data, y.data))
        return joined


join = Join()


class SlideBase(Access):
    def _preprocess(self, input):
        if isinstance(input, FractalTensor):
            return input
        elif isinstance(input, Iterative):
            assert len(input)

            length = len(input)
            num_input = len(input[0])

            xs = []
            for i in range(num_input):
                items = []
                for j in range(length):
                    items.append(input[j][i])
                if isinstance(items[0], Tensor):
                    v = FractalTensor.from_tensors(*items)
                    pass
                elif isinstance(items[0], FractalTensor):
                    v = FractalTensor.from_fractaltensors(*items)
                    pass
                else:
                    raise TypeError()
                xs.append(v)
        else:
            raise TypeError(('Expected FractalTensor or Iterative, '
                             'got f{type(input).__name__}'))
        return xs


class Slide(SlideBase):
    def _slide_impl(self,
                    input: FractalTensor[T],
                    window_size: int,
                    stride: int,
                    dilation: int,
                    padding: int,
                    padding_value: Union[FractalTensor, Tensor] = None
                    ) -> FractalTensor[FractalTensor[T]]:
        if not (isinstance(padding_value, Tensor)
                or isinstance(padding_value, FractalTensor)):
            raise TypeError(('Expected Tensor or FractalTensor, '
                             f'got {type(padding_value).__name__}.'))
        assert padding_value._type.is_equal_type(input.element_type)
        paddings = [copy.deepcopy(padding_value) for _ in range(padding)]
        if isinstance(padding_value, Tensor):
            padding_seq = FractalTensor.from_tensors(*paddings)
        else:
            padding_seq = FractalTensor.from_fractaltensors(*paddings)

        padded_inputs = kaleido.frontend.operations.join(
            kaleido.frontend.operations.join(padding_seq, input), padding_seq)

        window_span = (window_size - 1) * (dilation - 1) + window_size
        end_pos = len(padded_inputs) - window_span + 1
        window_pos = list(range(0, end_pos, stride))

        slides = []
        for start_pos in window_pos:
            items_in_window = padded_inputs[start_pos:start_pos +
                                            window_span:dilation]
            slides.append(items_in_window)
        slides.append(padded_inputs[window_pos[-1] + 1::dilation])
        rv = FractalTensor.from_fractaltensors(*slides)
        return rv

    def __call__(self,
                 input: FractalTensor[T],
                 window_size: int,
                 stride: int,
                 dilation: int,
                 padding: int = None,
                 padding_value: T = None) -> FractalTensor[FractalTensor[T]]:
        """
        The formula to compute the output size of striding:
            L = $L_{\text{out}} = \lfloor \frac{L_{\text{in}} + 2 \times
                    \text{padding} - \text{dilation} \times
                    (\text{window_size} - 1) -1 }{\text{stride}} \rfloor$
        Args:
            input, FractalTensor,
            window_size, int,
            stride, int,
            dilation, int,
            padding, int,
            padding_value, Union[FractalTensor, Tensor]

        Returns:
            returned a FractalTensor with an increased depth by 1.
        """
        if window_size % 2 == 0:
            raise ValueError(
                'Even number of window size is not implemented yet.')
        if isinstance(input, FractalTensor):
            return self._slide_impl(input, window_size, stride, dilation,
                                    padding, padding_value)
        else:
            raise TypeError(
                f'Expected FractalTensor, got {type(input).__name__}.')


slide = Slide()


class ShiftedSlide(SlideBase):
    def _shift_slide_impl(self, input: FractalTensor[T], window_size: int,
                          dilation: int) -> FractalTensor[FractalTensor[T]]:
        window_span = (window_size - 1) * (dilation - 1) + window_size
        half_window = window_span // 2

        def _ids(x, L):
            if x < 0:
                return x + L - 1
            elif x >= L:
                return x - L
            else:
                return x

        L = input.length
        slides = []
        for pos in range(0, L):
            indices = list(
                map(lambda x: _ids(x, L),
                    range(pos - half_window, pos + half_window + 1, dilation)))
            slides.append(input[indices])
        rv = FractalTensor.from_fractaltensors(*slides)
        return rv

    def __call__(self,
                 input: Union[FractalTensor[T], Iterative],
                 window_size: int,
                 dilation: int = 1) -> FractalTensor[FractalTensor[T]]:
        if window_size % 2 == 0:
            raise ValueError(
                'Even number of window size is not implemented yet.')

        xs = self._preprocess(input)

        rvs = []
        for x in xs:
            v = self._shift_slide_impl(x, window_size, dilation)
            rvs.append(v)

        return rvs


shifted_slide = ShiftedSlide()


class Window(Access):
    def __call__(self,
                 index: Union[int, Tensor],
                 input: FractalTensor[T],
                 window_size: int,
                 dilation: int,
                 padding_value: T = None) -> FractalTensor[T]:
        """
        Retrieve elements falling in a window that is centred on the i-th
        element of input.
        """
        if not isinstance(input, FractalTensor):
            raise TypeError(
                f'Expected FractalTensor, got {type(input).__name__}.')
        assert padding_value._type.is_equal_type(input.element_type)

        index = index
        if isinstance(index, Tensor):
            assert index.numel == 1
            index = int(index.data)

        if index >= input.length:
            raise IndexError(f'{index} exceeds length of the input.')

        window_span = (window_size - 1) * (dilation - 1) + window_size
        half_window = window_span // 2

        left_pad = 0 if index - half_window > 0 else half_window - index
        right_pad = (0 if index + half_window < input.length else
                     index + half_window - input.length)

        padded_inputs = input
        if left_pad > 0:
            paddings = [copy.deepcopy(padding_value) for _ in range(left_pad)]
            padding_seq = FractalTensor.from_tensors(*paddings)
            padded_inputs = kaleido.frontend.operations.join(
                padding_seq, input)
        if right_pad > 0:
            paddings = [copy.deepcopy(padding_value) for _ in range(right_pad)]
            padding_seq = FractalTensor.from_tensors(*paddings)
            padded_inputs = kaleido.frontend.operations.join(
                input, padding_seq)

        end_pos = index + window_span
        start_pos = index - half_window

        if start_pos > 0:
            start_pos = 0
        items_in_window = padded_inputs[start_pos:start_pos +
                                        window_span:dilation]
        if isinstance(input.element_type, TensorStorage):
            return FractalTensor.from_tensors(*items_in_window)
        elif isinstance(input.element_type, FractalTensorStorage):
            return FractalTensor.from_fractaltensors(*items_in_window)
        else:
            raise TypeError()


window = Window()


class ShiftedWindow(Access):
    def __call__(self, index: Union[int, Tensor], input: FractalTensor[T],
                 window_size: int, dilation: int) -> FractalTensor[T]:
        """
        Retrieve elements falling in a window that is centred on the i-th
        element of input.

        window_span = (window_size - 1) * (dilation - 1) + window_size
        """
        if not isinstance(input, FractalTensor):
            raise TypeError(
                f'Expected FractalTensor, got {type(input).__name__}.')

        index = index
        assert index.numel == 1
        if isinstance(index, Tensor):
            index = int(index.data)

        if index >= input.length:
            raise IndexError(f'{index} exceeds length of the input.')

        window_span = (window_size - 1) * (dilation - 1) + window_size
        half = window_span // 2

        indices = [(i + input.length) % input.length
                   for i in range(index - half, index + half + 1, dilation)]
        items_in_window = input[indices]
        if isinstance(input.element_type, TensorStorage):
            return FractalTensor.from_tensors(*items_in_window)
        elif isinstance(input.element_type, FractalTensorStorage):
            return FractalTensor.from_fractaltensors(*items_in_window)
        else:
            raise TypeError()


shifted_window = ShiftedWindow()
