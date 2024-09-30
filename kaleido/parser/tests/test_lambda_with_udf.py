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
def udf1(x: Tensor['5, 17', float, 'cpu'],
         y: Tensor['5, 17', float, 'cpu']) -> Tensor['5, 17', float, 'cpu']:
    v1 = x + y
    v2 = ops.sigmoid(v1)
    return v2


@kaleido.function(ctx1)
def f1(xs: FractalTensor[Tensor['5, 17', float, 'cpu']]
       ) -> FractalTensor[Tensor['5, 17', float, 'cpu']]:
    zs = ops.scan(
        lambda s, x: udf1(s, x),
        xs,
        initializer=ops.zeros(shape=(5, 17), device='cpu', dtype='float'))
    return zs


ctx2 = kaleido.Context()


@kaleido.function(ctx2)
def udf2(a: Tensor['1, 128', float, 'cpu'], b: Tensor['1, 128', float, 'cpu'],
         c: Tensor['1, 128', float, 'cpu']) -> Tensor['1, 128', float, 'cpu']:
    c = a + b + c
    return c


@kaleido.function(ctx2)
def f2(xs: FractalTensor[Tensor['1, 128', float, 'cpu']]
       ) -> FractalTensor[Tensor['1, 128', float, 'cpu']]:
    ys = ops.scan(
        lambda s, x: udf2(x, s, s),
        xs,
        initializer=ops.zeros(shape=(1, 128), device='cpu', dtype='float'))
    return ys


ctx3 = kaleido.Context()


@kaleido.function(ctx3)
def udf3(a: Tensor['1, 128', float, 'cpu'], b: Tensor['1, 128', float, 'cpu'],
         c: Tensor['1, 128', float, 'cpu'],
         d: Tensor['1, 128', float, 'cpu']) -> Tensor['1, 128', float, 'cpu']:
    c = ops.sigmoid(ops.tanh(a + b) * c)
    d = c * d
    return d


@kaleido.function(ctx3)
def f3(xs: FractalTensor[Tensor['1, 128', float, 'cpu']],
       ys: FractalTensor[Tensor['1, 128', float, 'cpu']],
       z: Tensor['128, 64', float, 'cpu']
       ) -> FractalTensor[Tensor['1, 128', float, 'cpu']]:
    zs = ops.scan(
        lambda s, x: udf3(*x, s, z),
        ops.zip(xs, ys),
        initializer=ops.zeros(shape=(1, 128), device='cpu', dtype='float'))
    return zs


class TestParallelPatterns(unittest.TestCase):
    def setUp(self):
        L1 = 11
        L2 = 13

        self.x = Tensor((1, 128), kaleido.float32, device='cpu')
        self.x.initialize(torch.rand, *self.x.shape)

        self.xs1 = create_fractaltensor((5, 17), L1)
        self.xs2 = create_fractaltensor((1, 128), L1)

    def test1(self):
        yss = f1(self.xs1)
        ir = ctx1[-1].ir_block
        ir.propagate_storage()

        self.assertEqual(len(ir.input_ports), 1)
        self.assertEqual(len(ir.output_ports), 1)
        self.assertEqual(len(ir.nodes), 2)

        p = PlotProgram('figures/udf1')
        p.plot(ir)

    def test2(self):
        f2(self.xs2)

        ir = ctx2[-1].ir_block
        ir.propagate_storage()

        p = PlotProgram('figures/udf2')
        p.plot(ir)

    def test3(self):
        f3(self.xs2, self.xs2, self.x)

        ir = ctx3[-1].ir_block
        ir.propagate_storage()

        p = PlotProgram('figures/udf3')
        p.plot(ir)


if __name__ == '__main__':
    unittest.main()
    print_ast(f1)
