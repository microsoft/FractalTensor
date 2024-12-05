from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
from collections import OrderedDict

from kaleido.frontend.types import Storage
from kaleido.frontend.types import TensorStorage
from kaleido.frontend.types import FractalTensorStorage
from kaleido.parser.errors import ShapeError
from kaleido.parser.ir_nodes import OperationNode
from kaleido.parser.ir_nodes import Elementwise
from kaleido.parser.operations.common import registers
from kaleido.frontend.types import str_to_internal_type


@registers.tensor_primitives.register
class MatMult(OperationNode):
    opcode = 'matmul'
    arity = 2

    def __init__(self, name: str):
        super(MatMult, self).__init__(name, OrderedDict(), OrderedDict())

    def infer_shape(self, shape1: Tuple[int],
                    shape2: Tuple[int]) -> Tuple[int]:
        assert len(shape1) == 2
        assert len(shape2) == 2
        assert shape1[1] == shape2[0]

        output_shape = [0] * 2
        output_shape[0] = shape1[0]
        output_shape[1] = shape2[1]
        return output_shape

    def propagate_storage(self) -> Storage:
        super().propagate_storage()

        s1, s2 = self.input_ports.values()
        out_storage = s2.clone()
        out_storage.shape = self.infer_shape(s1.shape, s2.shape)
        self.output_ports[list(self.output_ports.keys())[-1]] = out_storage


@registers.tensor_primitives.register
class Zeros(OperationNode):
    """Constant operation."""

    opcode = 'zeros'
    arity = 0

    def __init__(self, name: str):
        super(Zeros, self).__init__(name, OrderedDict(), OrderedDict())

    def propagate_storage(self, *storages):
        super().propagate_storage()

        assert 'shape' in self.attributes
        assert 'device' in self.attributes
        assert 'dtype' in self.attributes

        storage = TensorStorage(self.attributes['shape'],
                                str_to_internal_type(self.attributes['dtype']),
                                self.attributes['device'])

        self.output_ports[list(self.output_ports.keys())[-1]] = storage


@registers.tensor_primitives.register
class Ones(OperationNode):
    """Constant operation."""

    opcode = 'ones'
    arity = 0

    def __init__(self, name: str):
        super(Ones, self).__init__(name, OrderedDict(), OrderedDict())


@registers.tensor_primitives.register
class Add(Elementwise):
    opcode = 'add'
    arity = 2

    def __init__(self, name: str):
        super(Add, self).__init__(name, OrderedDict(), OrderedDict())


@registers.tensor_primitives.register
class Tanh(Elementwise):
    opcode = 'tanh'
    arity = 1

    def __init__(self, name: str):
        super(Tanh, self).__init__(name, OrderedDict(), OrderedDict())


@registers.tensor_primitives.register
class Sigmoid(Elementwise):
    opcode = 'sigmoid'
    arity = 1

    def __init__(self, name: str):
        super(Sigmoid, self).__init__(name, OrderedDict(), OrderedDict())


@registers.tensor_primitives.register
class Mult(Elementwise):
    opcode = 'mult'
    arity = 2

    def __init__(self, name: str):
        super(Mult, self).__init__(name, OrderedDict(), OrderedDict())


@registers.tensor_primitives.register
class Softmax(OperationNode):
    opcode = 'softmax'
    arity = 1

    def __init__(self, name: str):
        super(Softmax, self).__init__(name, OrderedDict(), OrderedDict())


@registers.tensor_primitives.register
class Cross_Entropy(OperationNode):
    opcode = 'cross_entropy'
    arity = 2

    def __init__(self, name: str):
        super(Cross_Entropy, self).__init__(name, OrderedDict(), OrderedDict())


@registers.tensor_primitives.register
class Cat(OperationNode):
    opcode = 'cat'
    arity = -1  #

    def __init__(self, name: str):
        super(Cat, self).__init__(name, OrderedDict(), OrderedDict())
        self.attributes['dim'] = -1


@registers.tensor_primitives.register
class Split(OperationNode):
    opcode = 'split'
    arity = 1  #

    def __init__(self, name: str):
        super(Split, self).__init__(name, OrderedDict(), OrderedDict())
        self.attributes['num'] = -1
        self.attributes['dim'] = -1
