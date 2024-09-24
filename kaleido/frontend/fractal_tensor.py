from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Generic

import copy
from collections.abc import Sequence

import kaleido
from kaleido.frontend.tensor import Tensor
from kaleido.frontend.types import Number
from kaleido.frontend.types import TensorStorage
from kaleido.frontend.types import FractalTensorStorage
from kaleido.frontend.types import pytype_to_internal_type

__all__ = [
    'FractalTensor',
    'StaticList',
    'Iterative',
]

T = TypeVar('Tensor')
FT = TypeVar('FractalTensor')
IT = TypeVar('Iterative')

DType = TypeVar('DType')
Length = TypeVar('Length')


class StaticList(Sequence, Generic[DType, Length]):
    """
    StaticList has no implementation. It is used to annotate length of a list,
    and the length will not be changed after declaration.

    Usage:
        class ModelParams(NamedTuple):
            embedding: Tensor['1, 64', float, 'cpu']
            block_params: StaticList[BlockParams, '2']

    In Python3.9, `Annotated` can be used to annotate any meta information,
    like:

        from typing import Annotated
        block_params: Annotated[BlockParams, 2]
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(f'Cannot instantiate type {type(self)}')


class FractalTensor(Sequence, Generic[T]):
    """FractalTensor is a one-dimensional list-like contrainer of hemogenous tensors.

    FractalTensor could have variable length and could be nested.

    Constraints of FractalTensor:
    1. FractalTensor is a symbolic value in forward computation so that to support
       automatic differentiation. This means each position of FractalTensor can
       only be write once to fill in the value. This constraints is guaranteed
       by meta access operations for FractalTensor. Array access may lead to alias
       in program, this constrain will simplify alias removal and make program
       analysis more tractable.
    2. Tensors stored in FractalTensor MUST be homogenous in terms of having the
       same shape and elementary type, which will be interpreted as the smallest
       unit of addressable memory hold by a FractalTensor.
    3. Writing a FractalTensor by indices is prohibited (inheriting from
       Sequence instead of MutableSequence).

       Writing by indices could makes it abitrarily complicated to guarantee
       FractalTensor a symbolic in forward computation, and also, could make
       memory-based data dependence analysis abitrarily complicated.

       s1: x = a[i]
       s2: a[j] = y
       s3: z = a[i]

       whether x and z have a same value dependes on whehter i == j. Without
       constraints, subscript expression here could be a complicated expression,
       making determine whether s1 (read access) and s2 (write access) access
       the same data point an arbitariliy complicated constrained optimization
       problem looking only for integer solutions.
    """

    def __init__(self,
                 dtype: Union[Number, TensorStorage, FractalTensorStorage]):
        """
        Args:
            dtype, Union[Number, TensorStorage, FractalTensorStorage], the type of
                   elements stored in a FractalTensor.
        """
        self._type = FractalTensorStorage(dtype)

        # buffer that is a flatten list.
        self._T = None

    @property
    def shape(self) -> int:
        if self.depth != 0:
            raise ValueError('shape is undefined for a nested tensor array')
        return (self.flatten_shape
                if self.numel > 1 else self._type.element_shape())

    @property
    def device(self):
        dtype = self._type
        while not isinstance(dtype, TensorStorage):
            dtype = dtype._dtype
        return dtype._device

    @property
    def element_type(self):
        """Returns the element types of this FractalTensor."""
        return self._type.element_type()

    @property
    def element_shape(self) -> Tuple[int]:
        """Returns the element shape of this FractalTensor."""
        return self._type.element_shape()

    @property
    def element_size(self) -> int:
        """Returns the numel of elements stored in this FractalTensor."""
        return self.element_type.numel

    def _numel(self, shape: List) -> int:
        """Count numel of the give subtree."""
        n = 0
        if isinstance(shape, int):
            n += 1
            return n
        else:
            for s in shape:
                n += self._numel(s)
            return n

    @property
    def numel(self) -> int:
        """Returns number of tensors stored in the FractalTensor.

        This attribute only has value after a FractalTensor is initialized at
        runtime.
        """
        if self._type.indices is None:
            raise RuntimeError('FractalTensor is not initialized.')
        return self._numel(self._type.indices)

    @property
    def flatten_shape(self) -> Tuple[int]:
        """Returns the flatten shape of a nested tensor array."""
        return (self.numel, ) + tuple(self._type.element_shape())

    @property
    def length(self) -> int:
        """This attribute only has valid value during runtime."""
        return self._type.length

    @property
    def depth(self) -> int:
        return self._type.depth

    @property
    def data(self):
        return self._T

    @data.setter
    def data(self, v):
        # NOTE: set self._T use data.setter. `setter` does necessary checks.
        # Do not directly modify set self._T
        # self._T is regarded as a buffer, erase its origianl size.
        self._T = v.view(-1)

    @property
    def indices(self):
        return self._type._indices

    @indices.setter
    def indices(self, value: List):
        # NOTE: use setter instead of directly modifying self.indices
        # since `setter` does necessary check.
        self._type.indices = value

    def width_by_depth(self, depth: int) -> List[int]:
        """Return at the given depth, how many children each sub-tree has.

        Args:
            depth, int, the innermost elements have the smallest depth 0, and
                   the outermost elements have the deepest depth.
                   A negative integer can be assigned to `depth` which will be
                   interperted as `self.depth + depth + 1`. `depth` has a range
                   from [1, self.depth].

        NOTE: results of this function is data-dependent, therefore, this
        function is ONLY called and meanfuling at runtime.
        """

        depth_ = depth if depth >= 0 else self.depth + depth + 1
        if depth_ > self.depth:
            raise ValueError(f'depth exceed {self.depth}.')
        if depth_ == 0:
            raise ValueError('width is not defined for depth 0.')

        if self.indices is None:
            raise RuntimeError(
                'width_by_depth is data dependent. No data is given.')

        def _bfs(root: List, target_depth: int):
            cur_depth = 0
            to_visit: List = root
            while to_visit:
                if cur_depth == target_depth:
                    return [len(x) for x in to_visit]

                width = len(to_visit)
                for i in range(width):
                    top = to_visit.pop(0)

                    if isinstance(top, List):
                        to_visit += [item for item in top]
                    else:
                        raise RuntimeError('depth is out of range.')
                cur_depth += 1

        return _bfs([copy.deepcopy(self.indices)], self.depth - depth_)

    def __add__(self, y: T) -> T:
        if self.data is not None and y.data is not None:
            if self.numel != y.numel or self.data.shape[0] != y.data.shape[0]:
                raise RuntimeError(
                    'Fail to add x and y due to their inconsistent shape.')

            ta = FractalTensor(
                TensorStorage(self.element_shape, self.element_type._dtype))
            ta.data = self.data + y.data
            ta.indices = self._type.indices
            return ta
        return None

    def is_equal_type(self, y: FT) -> bool:
        if not isinstance(y, FractalTensor):
            return False

        return self._type.is_equal_type(y._type)

    def initialize(self, initializer: callable, *args, **kwargs) -> None:
        self.data = initializer(*args, **kwargs)

    def index(self, idx: int) -> FT:
        """Retrive the idx-th element from the FractalTensor.

        Internally, nested FractalTensor form a tree-form addressing process.

        Args:
            idx, int

        Returns:
            FractalTensor
        """
        if self.data is None:
            raise RuntimeError("FractalTensor is not initalized.")

        start = 0
        i = 0
        while i < idx:
            start += self._numel(self.indices[i])
            i += 1
        end = start + self._numel(self.indices[idx])
        if end > self.numel:
            raise Exception("Out of boundary.")

        v = self.data[start * self.element_size:end * self.element_size]

        dtype = self._type._dtype  # elementary type of the current tensor array
        if isinstance(dtype, FractalTensorStorage):
            dtype = dtype._dtype
        elif isinstance(dtype, TensorStorage) or isinstance(dtype, Number):
            dtype = dtype  # degrade to a tensor/scalar.
        else:
            raise TypeError('FractalTensor type construtor error.')

        if self.depth > 1:
            x = FractalTensor(dtype)
            x.indices = self.indices[idx]
        else:
            if isinstance(dtype, TensorStorage):
                x = Tensor(dtype.shape, dtype._dtype, device=dtype._device)
            else:
                raise NotImplementedError()

        x.data = v.view([1] + [int(x) for x in v.shape])
        if self.depth == 1:
            # A tensor is returned
            x.data = x.data.view(self.element_shape)
        return x

    def first(self) -> FT:
        return self.index(0)

    def last(self) -> FT:
        return self.index(self.length - 1)

    def join(self, x) -> FT:
        return kaleido.frontend.operations.join(self, x)

    def _multiple_ids(self, indices):
        if isinstance(self.element_type, TensorStorage) and self.depth == 1:
            return FractalTensor.from_tensors(
                *[self.index(ids) for ids in indices])
        else:
            return FractalTensor.from_fractaltensors(
                *[self.index(ids) for ids in indices])

    def __getitem__(self, i):
        if type(i) is slice:
            start = i.start if i.start is not None else 0
            stop = i.stop

            if i.stop is None:
                stop = len(self.indices)
            if stop < 0:
                stop = stop + len(self.indices)

            step = i.step if i.step is not None else 1
            ids = list(range(start, stop, step))

            if isinstance(self.element_type,
                          TensorStorage) and self.depth == 1:

                return FractalTensor.from_tensors(
                    *[self.index(i) for i in ids])
            else:
                return FractalTensor.from_fractaltensors(
                    *[self.index(i) for i in ids])
        elif isinstance(i, List):
            if isinstance(i[0], int):
                return self._multiple_ids(i)
            else:
                raise NotImplementedError()
        elif isinstance(i, Tensor):
            if not isinstance(i.element_type, kaleido.frontend.types.Int):
                raise TypeError('IntTensor is expected.')
            v = self._multiple_ids(i.data.tolist()[0])
            return v
        else:
            return self.index(i)

    def __len__(self) -> int:
        return self._type.length

    def __str__(self):
        return str(self._type)

    __repr__ = __str__

    @staticmethod
    def from_fractaltensors(*arrays: FT) -> FT:
        """Create a nested FractalTensor from a tuple of FractalTensors.

        Returns:
            a FractalTensor with an increased depth by 1.
        """
        head = arrays[0]

        # FractalTensor is initialized at runtime. It has values.
        check_data = (head.data is not None)

        for ids, array in enumerate(arrays[1:]):
            if not head._type._dtype.is_equal_type(array._type._dtype):
                raise ValueError((
                    "Elements stored in FractalTensor should be homogenous:\n"
                    "Inconsistent element type:\n"
                    f"{str(head._type._dtype)} vs.\n{str(array._type._dtype)}.\n"
                ))
            if head.depth != array.depth:
                raise ValueError(
                    ("Elements stored in FractalTensor should be homogenous:\n"
                     "Inconsistent element types: inconsistent depth : "
                     f"{str(head.depth)} vs. {str(array.depth)}."))

            if check_data and array.data is None:
                raise RuntimeError(
                    f"{ids}-th FractalTensor is not initalized.")

        x = FractalTensor(head._type)
        x._type._depth = head.depth + 1

        if check_data:
            x.indices = [array.indices for array in arrays]
            x.data = kaleido.frontend.operations.cat(
                [Tensor.from_tensor(array.data) for array in arrays],
                dim=0).data.view(-1)
        return x

    @staticmethod
    def from_tensors(*xs: Tensor) -> FT:
        """Create a 1-depth FractalTensor from list of tensor."""

        if not xs:
            raise ValueError("input should be a non-empty list.")

        head = xs[0]

        for x in xs:
            if not isinstance(x, Tensor):
                raise ValueError(f'Expected tensor, got {type(x).__name__}.')

            if not head._type.is_equal_type(x._type):
                raise ValueError(
                    ("Elements stored in FractalTensor should be homogenous:\n"
                     "Inconsistent element type: "
                     f"{str(head._type)} vs. {str(x._type)}.\n"))

        x = FractalTensor(head._type)
        x.data = kaleido.frontend.operations.stack(xs, dim=0).data.view(-1)
        x.indices = list(range(len(xs)))
        return x

    @staticmethod
    def from_pylist(x: List, device='cpu') -> FT:
        """Convert a nested Python list into a nested FractalTensor.

        Args:
            x, List[T], T = Union[List, np.ndarray, int]

        Extract type information for nested FractalTensor from nested Python list.
        """

        def _bfs(x: List):
            """
            Returns:
                data: List, a depth-1 Python list that is the flatten structure of
                      original nested list. It stores all values in the original
                      nested.
            """
            to_visit: List = copy.deepcopy(x)  # FIXME(ying): avoid deep copy.
            data = []
            lens = []

            while to_visit:
                width = len(to_visit)
                lens.append([])
                for i in range(width):
                    top = to_visit.pop(0)

                    if isinstance(top, List):
                        lens[-1].append(len(top))
                    else:
                        if isinstance(top, int):
                            lens[-1].append(1)

                        data.append(top)

                    if isinstance(top, List):
                        to_visit += [item for item in top]
            del lens[-1]  # the last element is an empty list.

            return lens, data

        def _gen_indices(lens: List):
            assert len(lens) == depth - 1

            flatten_ids = []
            for seq_len in lens[-1]:
                flatten_ids.append(list(range(seq_len)))

            cur_ids = []
            prev_ids = flatten_ids
            for depth_lens in reversed(lens[:-1]):
                assert sum(depth_lens) == len(prev_ids)
                start = 0
                for seq_len in depth_lens:
                    end = start + seq_len
                    cur_ids.append(prev_ids[start:end])
                    start = end
                prev_ids = cur_ids
            return cur_ids if cur_ids else prev_ids

        if not isinstance(x, List):
            raise RuntimeError('x should be a python list.')

        elem = x
        depth = 0
        while isinstance(elem, List):
            elem = elem[0]
            depth += 1

        if not isinstance(elem, int):
            raise NotImplementedError()

        ta_type = FractalTensorStorage(
            TensorStorage((1, ), pytype_to_internal_type(elem), device=device))
        for i in range(depth - 1):
            ta_type = FractalTensorStorage(ta_type)

        ta = FractalTensor(ta_type)
        ta._type = ta._type._dtype  # FIXME(ying), hotfix

        import torch  # TODO(ying): a hard-code to transform python list into a
        # PyTorch Tensor.
        if isinstance(x, List) and isinstance(x[0], int):
            ta.data = torch.LongTensor(
                x) if device == 'cpu' else torch.cuda.LongTensor(x)
            ta_type.indices = [1] * len(x)
            return ta

        lens, data = _bfs(x)
        ta_type.indices = _gen_indices(lens)
        ta.data = torch.LongTensor(
            data) if device == 'cpu' else torch.cuda.LongTensor(data)
        return ta


class Iterative(Sequence, Generic[T]):
    def __init__(self, *xs):
        def _check_input(x):
            #TODO(ying): Not implemented.
            return x

        self._fields = [_check_input(x) for x in xs]

    def to_string(self, indent: int = 0):
        indent_str = indent * ' '
        type_str = '{}{}({})\n'.format(indent_str, self.__class__.__name__,
                                       len(self._fields))

        indent += 2
        indent_str = indent * ' '
        for idx, field in enumerate(self._fields):
            type_str += '{}|- field {} = {}\n'.format(indent_str, idx,
                                                      str(self._fields[idx]))
        return type_str

    def __str__(self):
        # TODO(ying): refine pretty print.
        return self.to_string()

    __repr__ = __str__

    def __getitem__(self, i: int):
        return self._fields[i]

    def __len__(self):
        return len(self._fields)

    @staticmethod
    def make_iterative(*xs: Union[FractalTensor, Tensor, IT]) -> IT:
        #TODO(ying): checking is not implemented.
        return Iterative(*xs)

    def unpack(self) -> Tuple:
        return [x for x in self.fields]
