from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps
import ast
import asttokens
import inspect
import astpretty
import textwrap
import collections

from typing import Dict, List

from kaleido.parser.errors import *
from kaleido.parser.context import Context
from kaleido.parser.ir_nodes import NodeBase, BlockNode
from kaleido.parser.ast_visitor import AstVisitor
from collections.abc import Sequence

__all__ = [
    'Context',
    'function',
    'params',
]


class Parser(object):
    def __init__(self, func, ctx: Context, start_line_no=0):
        source_ = inspect.getsource(func)
        self.func = func
        self.source = textwrap.dedent(source_)
        self.col_offset = len(source_.split("\n")[0]) - len(
            source_.split("\n")[0])

        _, file_lineno = inspect.getsourcelines(func)
        self.file_lineno = file_lineno

        self.ctx = ctx

    def parse_function(self):
        """
        Parse a user function which is decorated with @kaleido.function
        into a BlockNode.
        """

        tree = ast.increment_lineno(
            asttokens.ASTTokens(self.source, parse=True).tree,
            self.file_lineno)

        function_def = tree.body[0]
        if len(tree.body) > 1 and not isinstance(function_def,
                                                 ast.FunctionDef):
            raise ParseError("Parser is applied only to a single function.")
        assert len(tree.body) == 1

        visitor = AstVisitor(self.ctx)
        visitor.visit(function_def)

        return self.ctx[0].ir_block

    def parse_parameter(self):
        tree = ast.increment_lineno(
            asttokens.ASTTokens(self.source, parse=True).tree,
            self.file_lineno)
        class_def = tree.body[0]

        if len(tree.body) > 1 and not isinstance(class_def, ast.ClassDef):
            raise UnsupportedType(
                "Expected a class inheriting from NamedTuple.")

        assert len(class_def.bases) == 1
        if class_def.bases[0].id != 'NamedTuple':
            raise UnsupportedType(
                "Expected a class inheriting from NamedTuple.")

        visitor = AstVisitor(self.ctx)
        visitor.visit(class_def)


def function(context, *arg, **kwargs):
    """parse a user-defined functoin into an internal AST.
    """

    if not isinstance(context, Context):
        raise TypeError('Expected Context.')

    def decorator_function(func):
        if not func.__name__ in context._compiled:
            parser = Parser(func, context)
            parser.parse_function()
            context._compiled.add(func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator_function


def params(context, *args, **kwargs):
    if not isinstance(context, Context):
        raise TypeError('Expected Context.')

    class ClassWrapper:
        def __init__(self, cls):
            self.wrapped_class = cls

            parser = Parser(self.wrapped_class, context)
            parser.parse_parameter()

        def __call__(self, *cls_ars, **cls_kwargs):
            return self.wrapped_class(*cls_ars, **cls_kwargs)

    return ClassWrapper


from kaleido.parser.operations.common import import_all_modules_for_register
import_all_modules_for_register()
