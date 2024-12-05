import ast
import asttokens
import inspect
import astpretty
import textwrap

import torch

import kaleido
from kaleido import Tensor
from kaleido import FractalTensor
from kaleido.frontend.types import TensorStorage
from kaleido.frontend.types import FractalTensorStorage

__all__ = [
    'get_ast',
    'print_ast',
    'create_fractaltensor',
    'create_depth2_fractaltensor',
]


def get_ast(func):
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    col_offset = len(source.split("\n")[0]) - len(source.split("\n")[0])

    _, file_lineno = inspect.getsourcelines(func)

    return ast.increment_lineno(
        asttokens.ASTTokens(source, parse=True).tree, file_lineno)


def print_ast(func):
    astpretty.pprint(get_ast(func))


def create_fractaltensor(size, length):
    xs = FractalTensor(TensorStorage(size, kaleido.float32, device='cpu'))
    xs.indices = list(range(length))
    xs.initialize(torch.rand, *xs.flatten_shape)
    return xs


def create_depth2_fractaltensor(size, length1, length2):
    xss = FractalTensor(
        FractalTensorStorage(
            TensorStorage(size, kaleido.float32, device='cpu')))
    xss.indices = [list(range(length1)) for _ in range(length2)]
    xss.initialize(torch.rand, *xss.flatten_shape)
    return xss
