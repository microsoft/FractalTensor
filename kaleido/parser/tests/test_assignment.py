from __future__ import print_function

from context import *

from typing import Tuple
from typing import NamedTuple
from collections import OrderedDict

import torch

import kaleido
from kaleido import operations as ops
from kaleido import Tensor
from kaleido import FractalTensor
from kaleido.frontend.types import TensorStorage
from kaleido.frontend.types import FractalTensorStorage

from kaleido.parser.tests.utils import *
from kaleido.parser.plot import PlotProgram

ctx1 = kaleido.Context()


@kaleido.function(ctx1)
def f1(
        a: Tensor['128, 45', float, 'cpu'], b: Tensor['45, 128', float, 'cpu']
) -> Tuple[Tensor['128, 128', float, 'cpu'], Tensor['128, 128', float, 'cpu']]:
    c = a @ b
    d = ops.tanh(c)
    d = d + d
    return c, d


ctx2 = kaleido.Context()


@kaleido.function(ctx2)
def f2(a: Tuple[Tensor['128, 45', float, 'cpu'], Tensor[
        '45, 128', float, 'cpu']]) -> Tensor['128, 128', float, 'cpu']:
    x, y = a
    b = x @ y
    return b


ctx3 = kaleido.Context()


@kaleido.function(ctx3)
def f3(xs: FractalTensor[Tensor['1, 128', float, 'cpu']]
       ) -> Tensor['1, 128', float, 'cpu']:
    ys = ops.map(lambda x: ops.tanh(x) + ops.sigmoid(x), xs)
    ys = ys[:3]
    ys = ops.map(lambda x: ops.add(*x), ops.zip(ys, ys))
    y = ops.reduce(
        lambda s, x: s * x,
        ys,
        initializer=ops.zeros(shape=(1, 128), device='cpu', dtype='float'))
    return y


ctx4 = kaleido.Context()


@kaleido.function(ctx4)
def f4(xss: FractalTensor[FractalTensor[Tensor['1, 128', float, 'cpu']]]
       ) -> FractalTensor[FractalTensor[Tensor['1, 128', float, 'cpu']]]:
    yss = ops.map(lambda xs: ops.map(lambda x: ops.tanh(x), xs), xss)
    return yss


ctx5 = kaleido.Context()


@kaleido.function(ctx5)
def f5(xs: FractalTensor[Tensor['5, 17', float, 'cpu']]
       ) -> FractalTensor[Tensor['5, 17', float, 'cpu']]:
    ys = ops.map(lambda x: x + x, xs)
    return ys


ctx6 = kaleido.Context()


@kaleido.function(ctx6)
def f6(xss: FractalTensor[FractalTensor[Tensor['5, 17', float, 'cpu']]]
       ) -> FractalTensor[FractalTensor[Tensor['5, 17', float, 'cpu']]]:
    yss = ops.map(lambda xs:
            ops.scan(lambda s, x: s + x,
                xs,
                initializer=ops.zeros(shape=(5, 17), device='cpu', dtype='float')),
            xss)
    return yss


class TestAssignment(unittest.TestCase):
    M = 13
    N = 17

    def setUp(self):
        self.x = Tensor((128, 45), kaleido.float32, device='cpu')
        self.x.initialize(torch.rand, *self.x.shape)

        self.y = Tensor((45, 128), kaleido.float32, device='cpu')
        self.y.initialize(torch.rand, *self.y.shape)

        self.z = Tensor((1, 128), kaleido.float32, device='cpu')
        self.z.initialize(torch.rand, *self.z.shape)

        self.xs1 = create_fractaltensor((1, 128), TestAssignment.M)
        self.xss1 = create_depth2_fractaltensor((1, 128), TestAssignment.M,
                                                TestAssignment.N)

        self.xs2 = create_fractaltensor((5, 17), TestAssignment.M)
        self.xss2 = create_depth2_fractaltensor((5, 17), TestAssignment.M,
                                                TestAssignment.N)

    def test1(self):
        f1(self.x, self.y)

        ir = ctx1[0].ir_block
        ir.propagate_storage()

        self.assertEqual(len(ir.input_ports), 2)
        self.assertEqual(len(ir.output_ports), 2)
        self.assertEqual(len(ir.nodes), 3)

        p = PlotProgram('figures/assignment1')
        p.plot(ir)

    def test2(self):
        f2((self.x, self.y))
        ir = ctx2.peek().ir_block
        ir.propagate_storage()

        p = PlotProgram('figures/assignment2')
        p.plot(ir)

        self.assertEqual(len(ir.input_ports), 2)
        self.assertEqual(len(ir.output_ports), 1)
        self.assertEqual(len(ir.nodes), 1)

    def test3(self):
        f3(self.xs1)

        ir = ctx3.peek().ir_block
        ir.propagate_storage()

        p = PlotProgram('figures/assignment3')
        p.plot(ir)

    def test4(self):
        f4(self.xss1)

        ir = ctx4.peek().ir_block
        ir.propagate_storage()
        p = PlotProgram('figures/assignment4')
        p.plot(ir)

    def test5(self):
        ys = f5(self.xs2)

        ir = ctx5[0].ir_block
        ir.propagate_storage()

        self.assertEqual(len(ir.input_ports), 1)
        self.assertEqual(len(ir.output_ports), 1)
        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(len(ir.nodes['%node0'].nodes), 1)

        p = PlotProgram('figures/assignment5')
        p.plot(ir)

    def test6(self):
        yss = f6(self.xss2)

        ir = ctx6[0].ir_block
        ir.propagate_storage()

        self.assertEqual(len(ir.input_ports), 1)
        self.assertEqual(len(ir.output_ports), 1)
        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(len(ir.nodes['%node0'].nodes), 2)

        p = PlotProgram('figures/assignment6')
        p.plot(ir)


if __name__ == '__main__':
    unittest.main()
