"""All types considered in the computational process:

1. Basic arithmetic type: Number including float(16, 32, 64), integer(16, 32, 64), bool
2. Tensor type: Basic Arithmetic Type + TensorShape
3. FractalTensor type: List of Basic Arithmetic Type, Tensor or FractalTensor.
   - FractalTensor element should be homogenous.
   - The storage of a tensor array comes in blocks of contiguous bytes, where
     a byte is the smallest unit of addressable memory.
     - a byte is also a type that is either a basic arithmetic type, or a tensor
       type.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from anytree import NodeMixin
from anytree import RenderTree
from anytree import AsciiStyle
import copy
from collections.abc import Sequence
from collections import OrderedDict
from typing import Tuple
from typing import List
from typing import Union
from typing import TypeVar

import itertools
import functools
import operator

import torch

__all__ = [
    'float32',  # TODO(ying), singleton from metaclass
    'int32',  # TODO(ying), singleton from metaclass
    'int64',
    'Storage',
    'Number',
    'TensorStorage',
    'FractalTensorStorage',
    'StaticListStorage',
    'LiteralConstant',
    'pytype_to_internal_type',
    'internal_type_to_torch_type',
    'str_to_internal_type',
    'StorageInfoTree',
]


class Storage(object):
    # TODO(Ying): Implemented as a metaclass.

    def __init__(self, *args, **kwargs):
        raise RuntimeError(f'Cannot instantiate type {type(self)}')


class Number(Storage):
    def __init__(self, byte, signed):
        """Type constructor for basic type."""

        if byte not in [16, 32, 64]:
            raise ValueError("The byte should be 16, 32, or 64.")
        self.byte = byte
        self.signed = signed

    def to_string(self, indent: int, depth: int = 0):
        indent_str = indent * ' '

        if self.signed:
            return indent_str + 'signed ' + self.__class__.__name__ + str(
                self.byte)
        return indent_str + 'unsigned ' + self.__class__.__name__ + str(
            self.byte)

    def __str__(self):
        return self.to_string(0)

    __repr__ = __str__

    def is_equal_type(self, y):
        """Check type equivalence."""
        if type(self) != type(y):
            return False
        return (self.byte == y.byte) and (self.signed == y.signed)

    @property
    def shape(self):
        return self.element_shape()

    def element_shape(self) -> Tuple[int]:
        return (1, )

    def element_type(self):
        return type(self)


class Int(Number):
    def __init__(self, byte):
        super(Int, self).__init__(byte, signed=1)


class Real(Number):
    def __init__(self, byte):
        super(Real, self).__init__(byte, signed=1)

    @property
    def max(self):
        if self.byte == 32:
            return torch.finfo(torch.float32).max
        else:
            raise NotImplementedError()


class Bool(Number):
    def __init__(self):
        super(Bool, self).__init__(byte=64, signed=0)


# FIXME, for experiment only. Not well implemented.
float32 = Real(32)
int32 = Int(32)
int64 = Int(64)


def pytype_to_internal_type(x):
    if isinstance(x, int):
        return int64
    elif isinstance(x, float):
        return float32
    else:
        raise NotImplementedError()


def internal_type_to_torch_type(dtype):
    if dtype is float32:
        return torch.float32
    elif dtype is int32:
        return torch.int32
    else:
        raise NotImplementedError()


def str_to_internal_type(dtype: str):
    if dtype == 'float':
        return float32
    elif dtype == 'int':
        return int32
    else:
        raise NotImplementedError()


T = TypeVar('T')
FT = TypeVar('FT')


class StaticListStorage(Storage):
    def __init__(self, dtype: str, length: int):
        self.dtype = dtype
        self.length = length

    def __str__(self, indent: int = 0):
        indent_str = indent * ' '
        type_str = '{}{}\n'.format(
            indent_str, self.__class__.__name__.replace('Storage', ''))
        indent += 2
        indent_str = indent * ' '
        type_str += '{}|- dtype = {}\n'.format(indent_str, self.dtype)
        type_str += '{}|- length = {}\n'.format(indent_str, self.length)

        return type_str

    __repr__ = __str__


class TensorStorage(Storage):
    def __init__(self,
                 shape: Tuple[int],
                 dtype: Number,
                 device: str,
                 order: str = None,
                 strides: Tuple[int] = None):
        if not isinstance(dtype, Number):
            raise TypeError(
                'Element type of a tensor should be a Number type.')

        self._order = 'row' if order is None else order
        if self._order not in ['row', 'col']:
            raise ValueError(f'Unsuported layout: {order}.')

        if device is not None:
            if device not in ['cpu', 'cuda']:
                raise ValueError(f'Unsuported device: {device}.')

        self._shape = shape
        self._dtype = dtype
        self._device = device

        # Tensor is a dense and regularly shaped collection of scalars, whose
        # elements are contiguously lay out in physical memory.
        # strides is to translate a logical position into a location in
        # physical memory.
        self._strides = self._get_strides(strides)

    def _get_strides(self, strides) -> Tuple[int]:
        if strides is not None:
            if len(strides) != len(self._shape):
                raise ValueError('strides should have a same length as shape.')
            return strides
        else:
            if len(self._shape) == 1:
                return [1]
            else:
                strides = list(
                    itertools.accumulate([1] + list(self._shape[::-1]),
                                         operator.mul))[:-1][::-1]
                if self._order == 'row':
                    return strides
                elif self._order == 'col':
                    strides[-2] = 1
                    strides[-1] = self._shape[-2]
                    return strides
                else:
                    raise ValueError(f'Unknown memory layout {self._order}.')

    def recompute_strides(self):
        self._strides = self._get_strides(None)

    def clone(self):
        return copy.deepcopy(self)

    def to_string(self, indent: int = 0, depth: int = 0):
        indent_str = indent * ' '
        type_str = '{}{}\n'.format(
            indent_str, self.__class__.__name__.replace('Storage', ''))

        indent += 2
        indent_str = indent * ' '
        if depth == 0:
            type_str += '{}|- depth = 0\n'.format(indent_str)
        type_str += '{}|- shape = {}\n'.format(indent_str, str(self._shape))
        type_str += '{}|- strides = {}\n'.format(indent_str,
                                                 str(self._strides))
        type_str += '{}|- dtype = {}\n'.format(indent_str, self._dtype)
        type_str += '{}|- device = {}\n'.format(indent_str, str(self._device))
        type_str += '{}|- layout = {}\n'.format(indent_str, str(self._order))

        return type_str

    def __str__(self, indent: int = 0, depth: int = 0):
        return self.to_string(indent, depth)

    __repr__ = __str__

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, s):
        self._shape = s
        self.recompute_strides()

    @property
    def element_type(self):
        return self._dtype

    @property
    def numel(self) -> int:
        return functools.reduce(operator.mul, self._shape, 1)

    def _is_equal_shape(self, s: Tuple[int]):
        if len(self.shape) != len(s):
            return False

        for s1, s2 in zip(self.shape, s):
            if s1 != s2:
                return False
        return True

    def is_equal_type(self, y):
        """Check type equivalence."""

        if type(self) != type(y):
            return False

        return self._dtype.is_equal_type(y._dtype) and self._is_equal_shape(
            y.shape)


class FractalTensorStorage(Storage):
    def __init__(self, dtype: Union[Number, TensorStorage, FT]):
        """Type construtor of FractalTensor."""

        if not (isinstance(dtype, TensorStorage) or isinstance(dtype, Number)
                or isinstance(dtype, FractalTensorStorage)):
            raise TypeError(('Element type of a tensor array should be a '
                             'Number type, a Tensor, or a FractalTensor.'))

        # self._length and self._indices are data-dependent attributes that
        # are set when data is avaliable and could be changed per mini-batch.
        self._length = -1  # -1 implies uninitialized tensor array.

        # this attribute is set at the same time when length is changed by:
        # length's setter; tensor array's ctor, meta operations.
        # for a nested tensor array, self._indices forms a tree strucutre
        # that is used by the addressing function.
        self._indices = None

        self._depth = 0  # depth 0 implies a Tensor.
        if isinstance(dtype, TensorStorage) or isinstance(dtype, Number):
            self._depth = 1
        elif isinstance(dtype, FractalTensorStorage):
            self._depth = dtype.depth + 1
        else:
            raise TypeError('Unsupported elementary type for a FractalTensor.')

        self._dtype = dtype

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, v: int):
        if (self.indices is not None) or self.depth > 1:
            raise RuntimeError('Cannot set length of a nested tensor array.')

        if v <= 0:
            raise RuntimeError('FractalTensor length should be larger than 1.')
        self._indices = list(range(v))
        self._length = v

    @property
    def depth(self):
        return self._depth

    def element_shape(self) -> Tuple[int]:
        element_type = self._dtype
        while not (isinstance(element_type, Number)
                   or isinstance(element_type, TensorStorage)):
            element_type = element_type._dtype
        return element_type.shape

    def element_type(self) -> Union[TensorStorage, Number]:
        element_type = self._dtype
        while not (isinstance(element_type, Number)
                   or isinstance(element_type, TensorStorage)):
            element_type = element_type._dtype
        return element_type

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, value: List):
        # NOTE: use setter instead of directly modifying self.indices
        # since `setter` does necessary check.
        if not len(value):
            raise RuntimeError('value should be unempty list.')

        depth = 1
        x = value[0]
        while isinstance(x, List):
            x = x[0]
            depth += 1

        if depth != self.depth:
            raise RuntimeError(
                ("Wrong indices is given, `depth` of which is not consistent "
                 f"with FractalTensor's declaration: {depth} vs. {self.depth}."
                 ))

        x = value
        t = self
        while isinstance(x, List):
            t._length = len(x)
            x = x[0]
            if isinstance(t, Number):
                break
            t = t._dtype
        self._indices = value

    def is_equal_type(self, y):
        """Check type equivalence."""

        if type(self) != type(y):
            return False

        return self.depth == self.depth and self._dtype.is_equal_type(y._dtype)

    def to_string(self, indent: int = 0, depth: int = 0):
        indent_str = indent * ' '
        type_str = '{}{}\n'.format(
            indent_str, self.__class__.__name__.replace('Storage', ''))

        indent += 2
        indent_str = indent * ' '
        type_str += '{}|- depth = {}\n'.format(indent_str, str(self.depth))

        if depth == 0:
            type_str += '{}|- length = {}\n'.format(indent_str,
                                                    str(self.length))
        indent += 2
        type_str += '{}|- dtype:\n{}'.format(
            indent_str, (self._dtype.to_string(indent, depth + 1)))
        return type_str

    def __str__(self, indent: int = 0, depth: int = 0):
        return self.to_string(indent, depth)

    __repr__ = __str__


class LiteralConstant(Storage):
    """Used in the IR program"""

    def __init__(self, value, device=None):
        self._value = value
        self.device = device

    @property
    def value(self):
        # cannot be changed after creation.
        return self._value

    def __str__(self, indent: int = 0):
        indent_str = indent * ' '
        type_str = '{}{}\n'.format(indent_str, self.__class__.__name__)
        indent += 2
        type_str += '{}|- value: {}\n'.format(indent_str, str(self.value))
        return type_str

    __repr__ = __str__


class StorageInfoTree(NodeMixin, Sequence):
    def __init__(self, name, storage=None, parent=None):
        super(StorageInfoTree, self).__init__()
        self.name = name
        self.storage = storage

        self.parent = parent

    @property
    def numel(self):
        def _numel(root) -> int:
            if root.is_leaf:
                return 1
            n = 0
            for child in root.children:
                n += _numel(child)
            return n

        return _numel(self)

    @property
    def is_leaf(self):
        return not (self.storage == None)

    @property
    def flatten(self) -> List[Tuple[str, Storage]]:
        def _flatten(root, rvs, prefix):
            if root.is_leaf:
                rvs[prefix] = root.storage

            for child in root.children:
                _flatten(child, rvs, '{}{}'.format(prefix, child.name))

        rvs = OrderedDict()
        _flatten(self, rvs, prefix=self.name)
        return rvs

    def __getitem__(self, i):
        if self.storage:
            """The leaf node."""
            return self

        return self.children[i]

    def __len__(self):
        if self.storage:
            return 1
        return len(self.children)

    def __str__(self, indent=0):
        treestr = ''
        indent_str = indent * ' '
        for _, _, node in RenderTree(self, style=AsciiStyle()):
            indent = 2**node.depth
            indent_str = indent * ' '
            if node.storage is None:
                treestr += '{}{}(depth = {})\n'.format(indent_str, node.name,
                                                       node.depth)
            else:
                treestr += '{}{}(depth = {})\n{}'.format(
                    indent_str, node.name, node.depth,
                    node.storage.__str__(indent + 2))
        return treestr

    __repr__ = __str__
