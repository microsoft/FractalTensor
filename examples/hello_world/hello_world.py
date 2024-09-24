import context

from typing import NamedTuple

import kaleido
from kaleido import Tensor
from kaleido import FractalTensor
from kaleido import operations as ops

from examples.hello_world.utils import *
from kaleido.parser.plot import PlotProgram

ctx = kaleido.Context()


@kaleido.params(ctx)
class Params(NamedTuple):
    Ws: FractalTensor[Tensor['512, 512', float, 'cpu']]
    Us: FractalTensor[Tensor['512, 512', float, 'cpu']]


@kaleido.function(ctx)
def f3(a: Tensor['1, 512', float, 'cpu'], b: Tensor['1, 512', float, 'cpu'],
       c: Tensor['512, 512', float, 'cpu'],
       d: Tensor['512, 512', float, 'cpu']) -> Tensor['1, 512', float, 'cpu']:
    y = a @ c + b @ d
    return y


@kaleido.function(ctx)
def f2(xs: FractalTensor[Tensor['1, 512', float, 'cpu']],
       w: Tensor['512, 512', float, 'cpu'], u: Tensor['512, 512', float, 'cpu']
       ) -> FractalTensor[Tensor['1, 512', float, 'cpu']]:
    ys = ops.scan(
        lambda s, x: f3(x, s, w, u),
        xs,
        initializer=ops.zeros(shape=(1, 512), device='cpu', dtype='float'))
    return ys


@kaleido.function(ctx)
def f1(xs: FractalTensor[Tensor['1, 512', float, 'cpu']],
       Ws: FractalTensor[Tensor['512, 512', float, 'cpu']],
       Us: FractalTensor[Tensor['512, 512', float, 'cpu']]
       ) -> FractalTensor[FractalTensor[Tensor['1, 512', float, 'cpu']]]:
    yss = ops.scan(
        lambda state, x: f2(state, *x), ops.zip(Ws, Us), initializer=xs)
    return yss


@kaleido.function(ctx)
def f(xss: FractalTensor[FractalTensor[Tensor['1, 512', float, 'cpu']]],
      params: Params) -> FractalTensor[FractalTensor[FractalTensor[Tensor[
          '1, 512', float, 'cpu']]]]:
    ysss = ops.map(lambda xs: f1(xs, params.Ws, params.Us), xss)
    return ysss


block = ctx[-1].ir_block
block.propagate_storage()

p = PlotProgram()
p.plot(block)

if __name__ == '__main__':
    param = Params(Ws=Ws, Us=Us)

    ysss = f(xss, param)
