from __future__ import print_function

from context import *

from typing import Tuple, NamedTuple
from collections import OrderedDict
import torch

import kaleido
from kaleido import operations as ops
from kaleido import Tensor, FractalTensor
from kaleido.frontend.types import TensorStorage, FractalTensorStorage
from kaleido.parser.errors import *
from kaleido.parser.ast_visitor import FunctionDefVisitor

from kaleido.parser.tests.utils import get_ast


#========== valiad input and return arguments
def f0(x: Tuple[FractalTensor[Tensor['1, 24', float, 'cuda']], Tensor[
        '27, 56', float, 'cpu']], y: Tensor['1, 512', float, 'cpu']
       ) -> Tuple[Tensor['4, 5', float, 'cuda'], FractalTensor[FractalTensor[
           Tensor['1, 64', float, 'cuda']]]]:
    pass


def f1(xs: Tuple[FractalTensor[Tensor['2, 23', float, 'cpu']], Tuple[Tensor[
        '3, 13', float, 'cpu'], Tensor['3, 13', float, 'cpu']]]
       ) -> FractalTensor[Tensor['2, 23', float, 'cpu']]:
    pass


#========== valiad input and return arguments
def f2(x: float) -> float:
    pass


def f4(x):
    pass


def f3(x: Tuple[float, Tensor['1, 124', float, 'cpu']]):
    pass


# a valid program.
ctx1 = kaleido.Context()


@kaleido.params(ctx1)
class Type1(NamedTuple):
    a: Tensor['5000, 64', float, 'cpu']

    b: FractalTensor[FractalTensor[Tensor['512, 512', float, 'cpu']]]
    c: Tuple[Tensor['1, 128', float, 'cpu'], FractalTensor[FractalTensor[
        Tensor['512, 512', float, 'cpu']]]]


@kaleido.params(ctx1)
class Type2(NamedTuple):
    a: Type1  # Type1 must be defined before Type2
    b: FractalTensor[FractalTensor[Tensor['512, 512', float, 'cpu']]]


class TestParseFunDef(unittest.TestCase):
    def return_ir(self, f):
        ctx = kaleido.Context()
        visitor = FunctionDefVisitor(ctx)
        tree = get_ast(f)
        visitor.parse_func_decl(tree.body[0])

        return ctx.peek().ir_block

    def test0(self):
        """Valid functon."""

        ir = self.return_ir(f0)
        self.assertEqual(len(ir.input_ports), 3)
        self.assertEqual(len(ir.output_ports), 2)

        ir = self.return_ir(f1)
        self.assertEqual(len(ir.input_ports), 3)
        self.assertEqual(len(ir.output_ports), 1)

    def test1(self):
        """Invalid function.

        Exception is raised at decorating time instead of execution time.
        """
        try:
            self.return_ir(f2)
        except UnsupportedType:
            pass

        try:
            self.return_ir(f3)
        except UnsupportedType:
            pass

        try:
            self.return_ir(f4)

        except AnnotationError:
            pass

    def test2(self):
        """valid program."""

        param1 = Type1(a=0., b=0., c=0.)
        self.assertEqual(len(param1), 3)

        param2 = Type2(a=param1, b=0.)
        self.assertEqual(len(param2), 2)

    def test3(self):
        """invalid program.

        Exception is raised at decorating time instead of execution time.
        """
        try:

            @kaleido.params(kaleido.Context())
            class Type3(object):
                pass
        except UnsupportedType:
            pass


if __name__ == '__main__':
    unittest.main()
