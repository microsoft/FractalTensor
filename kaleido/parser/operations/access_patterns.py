from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
from collections import OrderedDict

from kaleido.frontend.types import Storage
from kaleido.frontend.types import TensorStorage
from kaleido.frontend.types import FractalTensorStorage
from kaleido.parser.operations.common import registers
from kaleido.parser.ir_nodes import AccessNode


@registers.access.register
class Index(AccessNode):
    opcode = 'index'
    arity = 1

    def __init__(self, name: str):
        super(Index, self).__init__(name, OrderedDict(), OrderedDict())

    def propagate_storage(self) -> Storage:
        super(Index, self).propagate_storage()
        ids = self.attributes['index']
        self.output_ports[list(self.output_ports.keys())[-1]] = list(
            self.input_ports.values())[0].element_type()


@registers.access.register
class Last(AccessNode):
    opcode = 'last'
    arity = 1

    def __init__(self, name: str):
        super(Last, self).__init__(name, OrderedDict(), OrderedDict())


@registers.access.register
class Slice(AccessNode):
    opcode = 'slice'
    arity = 1

    def __init__(self, name: str):
        super(Slice, self).__init__(name, OrderedDict(), OrderedDict())

    def propagate_storage(self) -> Storage:
        super(Slice, self).propagate_storage()

        lower = self.attributes['lower']
        step = self.attributes['step']
        upper = self.attributes['upper']

        s_in = list(self.input_ports.values())[0].element_type()
        s_out = FractalTensorStorage(s_in)
        s_out.indices = list(range((upper - lower) // step))
        self.output_ports[list(self.output_ports.keys())[-1]] = s_out


@registers.access.register
class Slices(AccessNode):
    opcode = 'slices'
    arity = 1

    def __init__(self, name: str):
        super(Slices, self).__init__(name, OrderedDict(), OrderedDict())

    def propagate_storage(self) -> Storage:
        super().propagate_storage()
