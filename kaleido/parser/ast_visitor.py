from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kaleido.parser.plot import PlotProgram

import ast
import copy
from abc import ABCMeta
from abc import abstractmethod
from typing import Union
from typing import List
from typing import Dict
from typing import Tuple
from collections import OrderedDict
from absl import logging
from astpretty import pprint

import kaleido
from kaleido.frontend.types import Storage
from kaleido.frontend.types import StorageInfoTree
from kaleido.frontend.types import TensorStorage
from kaleido.frontend.types import FractalTensorStorage
from kaleido.frontend.types import StaticListStorage
from kaleido.parser.context import ContextFrame
from kaleido.parser.context import NameRecord
from kaleido.parser.operations.common import registers
from kaleido.parser.ir_nodes import NodeBase
from kaleido.parser.ir_nodes import OperationNode
from kaleido.parser.ir_nodes import BlockNode
from kaleido.parser.ir_nodes import ParallelNode
from kaleido.parser.ir_nodes import ApplyToEach
from kaleido.parser.ir_nodes import Aggregate
from kaleido.parser.ir_nodes import EdgeEnd
from kaleido.parser.errors import AnnotationError
from kaleido.parser.errors import UnsupportedType
from kaleido.parser.errors import UnsupportedConstruct
from kaleido.parser.errors import UnknownPrimitiveOps
from kaleido.parser.errors import ParseError

__all__ = [
    'AstVisitor',
]


def _str_to_internal_type(s):
    if s == 'float':
        return kaleido.float32
    elif s == 'int':
        return kaleido.float32
    else:
        raise NotImplementedError


class AstVisitorBase(object):
    """ visitor class """

    def visit(self, node: ast.AST):
        method_name = 'visit_' + type(node).__name__
        method = getattr(self, method_name, None)
        if method is None:
            method = self.fallback

        return method(node)

    def fallback(self, node):
        raise RuntimeError(f'no such method visit_{type(node).__name__}')


class AstVisitor(AstVisitorBase):
    """Vistor pattern."""

    def __init__(self, ctx):
        """
        ctx, Context, the global parsing context.
        """
        super(AstVisitor, self).__init__()

        self.ctx = ctx

    @abstractmethod
    def process(self, node, parent=None):
        """Subclass must implement this method."""
        pass

    def parse_func_name(self, node) -> str:
        # FIXME(ying): a white list is required to identify whether a function
        # is a primitive operations supported in a backend.
        if isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            raise UnsupportedConstruct(
                f'unable to parse function name from type(node).__name__.')

    def _create_tensor_storage(self, node: ast.Subscript):
        """Create TensorStorage from ast.Subscript."""
        assert isinstance(node, ast.Subscript)

        type_info = node.slice.value.elts
        shape = [
            int(s) for s in type_info[0].s.rstrip().rstrip(',').split(',')
        ]  # shape

        dtype = _str_to_internal_type(type_info[1].id)  # dtype
        device = type_info[2].s  # device

        return TensorStorage(shape, dtype, device=device)

    def _create_fractaltensor_storage(self, node: ast.Subscript):
        """Create FractalTensorStorage from ast.Subscript."""
        assert isinstance(node, ast.Subscript)

        itr = node.slice
        depth = 0
        while not isinstance(itr.value, ast.Tuple):
            depth += 1
            itr = itr.value.slice

        type_info = itr.value.elts
        shape = [int(s)
                 for s in type_info[0].s.rstrip(',').split(',')]  # shape
        dtype = _str_to_internal_type(type_info[1].id)  # dtype
        device = type_info[2].s  # device

        ta_storage = FractalTensorStorage(
            TensorStorage(shape, dtype, device=device))
        for i in range(depth - 1):
            ta_storage = FractalTensorStorage(ta_storage)
        return ta_storage

    def _add_as_child(self, parent: StorageInfoTree, child: StorageInfoTree):
        for child in child.children:
            new_name = '@{}'.format('@'.join(child.name.split('@')[1:]))

            t = copy.deepcopy(child)
            t.name = new_name
            t.parent = parent

    def parse_type_annotation(self, annotation: Union[ast.Subscript, ast.Name],
                              parent: StorageInfoTree) -> StorageInfoTree:
        """
        Given type annotation of a single argument, parse it into Storage
        information.

        Args:
            annotation, ast.Subscript: type annotation for a function argument.
            parent, StorageInfoTree,

        Returns:
            StorageInfoTree
        """

        if not annotation:
            """
            when type annotation is missing, this branch is hit:
            1. type annotation is required for the outermost function that
               defines the interface of the full program.
            """
            raise AnnotationError(('input and output argumets are required to '
                                   'be annotated with their types.'))
            """
            # TODO(ying) not implmented.
            2. type information is infered for all innter functions enclosed in
               the outermost function.

            """

        if isinstance(annotation, ast.Subscript):
            if annotation.value.id == 'Tensor':
                parent.storage = self._create_tensor_storage(annotation)
            elif annotation.value.id == 'FractalTensor':
                parent.storage = self._create_fractaltensor_storage(annotation)
            elif annotation.value.id == 'Tuple':
                next_node = annotation.slice.value
                if isinstance(next_node, ast.Tuple):
                    for i, item in enumerate(next_node.elts):
                        child = StorageInfoTree('@{}'.format(i), None, parent)
                        if isinstance(item, ast.Subscript):
                            self.parse_type_annotation(item, child)
                        else:
                            raise UnsupportedType()

                elif isinstance(next_node, ast.Subscript):
                    self.parse_type_annotation(
                        next_node, StorageInfoTree('@0', None, parent))
                else:
                    raise UnsupportedConstruct()
            elif annotation.value.id == 'StaticList':
                dtype = annotation.slice.value.elts[0].id
                assert dtype in self.ctx.global_type_def
                length = int(annotation.slice.value.elts[1].s)
                parent.storage = StaticListStorage(dtype, length)
            else:
                raise UnsupportedType()
        elif isinstance(annotation, ast.Name):
            # user-defined types.
            arg_type_name = annotation.id
            if not (arg_type_name in self.ctx.global_type_def):
                raise UnsupportedType(arg_type_name)

            self._add_as_child(parent, self.ctx.global_type_def[arg_type_name])
        else:
            raise UnsupportedType()

        return parent

    def visit_ClassDef(self, node):
        ClassDefVisitor(self.ctx).process(node)

    def visit_FunctionDef(self, node):
        """
        A kaleido program begins with function definition.
        Entry point of recursive parsing.
        """
        logging.warning(type(node).__name__)
        return FunctionDefVisitor(self.ctx).process(node)

    def visit_Return(self, node):
        logging.warning(type(node).__name__)
        return ReturnVisitor(self.ctx).process(node)

    def visit_Assign(self, node):
        logging.warning(type(node).__name__)
        return AssignVisitor(self.ctx).process(node)

    def visit_Lambda(self, node):
        logging.warning(type(node).__name__)
        return LambdaVisitor(self.ctx).process(node)

    def visit_Call(self, node):
        logging.warning(type(node).__name__)
        return CallVisitor(self.ctx).process(node)

    def visit_BinOp(self, node):
        logging.warning(type(node).__name__)
        return BinOpVisitor(self.ctx).process(node)

    def visit_UnaryOp(self, node):
        logging.warning(type(node).__name__)

    def visit_Name(self, node):
        return node.id

    def _retrieve_tuple_or_list_elements(self, elts):
        elem_type = type(elts[0]).__name__

        for elem in elts[1:]:
            assert type(elem).__name__ == elem_type

        if elem_type == 'Num':
            return [e.n for e in elts]
        elif elem_type == 'Name':
            return [e.id for e in elts]
        else:
            raise NotImplementedError()

    def visit_Tuple(self, node):
        return self._retrieve_tuple_or_list_elements(node.elts)

    def visit_List(self, node):
        return self._retrieve_tuple_or_list_elements(node.elts)

    def visit_Str(self, node):
        assert isinstance(node, ast.Str)
        return node.s

    def visit_Num(self, node):
        logging.warning(f'visit {type(node).__name__}')
        return node.n

    def visit_Attribute(self, node):
        """Syntax that access a name alias."""
        logging.warning(type(node).__name__)
        return '{}@{}'.format(node.value.id, node.attr)

    def visit_Starred(self, node):
        """Syntax that creates name alias."""
        logging.warning(type(node).__name__)
        return node.value.id

    def visit_Slice(self, node):
        def _get_number(num):
            if not num:
                return None
            if not isinstance(num, ast.Num):
                raise NotImplementedError()
            return num.n

        lower = _get_number(node.lower)
        step = _get_number(node.step)
        return [
            lower if lower else 0,
            _get_number(node.upper), step if step else 1
        ]

    def visit_Index(self, node):
        if not isinstance(node.value, ast.Num):
            raise NotImplementedError()
        return node.value.n

    def visit_Subscript(self, node):
        def _create_access_op(opcode):
            op = registers.access[opcode]
            opnode = op(self.ctx.gen_op_name())
            opnode.add_input_port(self.ctx.gen_var_name(None, opnode, 'use'))
            opnode.add_output_port(self.ctx.gen_var_name(None, opnode, 'gen'))
            return opnode

        if not (isinstance(node.value, ast.Name) or isinstance(
                node.slice, ast.Slice) or isinstance(node.slice, ast.Index)):
            raise NotImplementedError()

        var_name = node.value.id

        block = self.ctx.peek().partially_parsed_blocks[-1]
        target_node, target_port = self.ctx.search_var_generation(var_name)

        assert len(target_node) == 1 and len(target_port) == 1
        target_node, target_port = target_node[0], target_port[0]

        if isinstance(node.slice, ast.Index):
            opnode = _create_access_op('index')
            opnode.attributes['index'] = self.visit(node.slice)
        elif isinstance(node.slice, ast.Slice):
            index = self.visit(node.slice)
            opnode = _create_access_op('slice')

            opnode.attributes['lower'] = index[0]
            opnode.attributes['upper'] = index[1]
            opnode.attributes['step'] = index[2]

        block.add_node(opnode)
        tail = (target_node, target_port)
        tip = (opnode, next(reversed(opnode.input_ports)))
        block.add_edge(
            tail,
            tip,
            edge_type='in' if target_port.endswith('bodyin') else None)
        return opnode


class FunctionDefVisitor(AstVisitor):
    """
    Parse the ast.FunctionDef node which is the ENTRY node of the AST of
    user's function. Function definition is NOT allowed to be enclosed inside
    another function.
    """

    def __init__(self, ctx):
        super(FunctionDefVisitor, self).__init__(ctx)

    def _parse_func_input(self, args: ast.arguments) -> List[StorageInfoTree]:
        return [
            self.parse_type_annotation(arg.annotation,
                                       StorageInfoTree(arg.arg, None, None))
            for arg in args.args
        ]

    def _parse_func_returns(self, return_annotation, return_stmt: ast.Return
                            ) -> Union[None, List[StorageInfoTree]]:
        if return_annotation is None:
            raise AnnotationError(('Type annotation for returned value '
                                   'of the function is required.'))

        return self.parse_type_annotation(
            return_annotation,
            StorageInfoTree(self.ctx.gen_tmp_name(), None, None))

    def _check(self):
        # at least a function should has a return statement
        assert len(self.node.body) > 1
        if isinstance(self.node.body[-1], ast.Return):
            if not isinstance(self.node.body[-1].value, ast.Name):
                raise ParseError("Return statement should return a variable.")
        else:
            raise ParseError(("The last statement of a function "
                              "should be a return statement."))

        if not isinstance(self.node.body[-1], ast.Return):
            raise ParseError(("The last statement of a user's function "
                              "should be a return statement."))

        for stmt in self.node.body[:-1]:
            if not isinstance(stmt, ast.Assign):
                raise ParseError(
                    f"Only assignment is allowed. Got {type(stmt)}")

        if len(self.node.decorator_list) > 1:
            raise NotImplementedError()

    def parse_func_decl(self, node):
        block = BlockNode(node.name, input_ports=None, output_ports=None)
        self.ctx.push(ContextFrame(node.name, block))
        self.ctx.peek().partially_parsed_blocks.append(block)

        # parse function definition
        input_arguments = self._parse_func_input(node.args)
        output_arguments = self._parse_func_returns(node.returns,
                                                    node.body[-1])

        for x in input_arguments:
            assert isinstance(x, StorageInfoTree)
            for name, storage in x.flatten.items():
                block.add_input_port(
                    self.ctx.gen_var_name(name, block, 'def'), storage)

        for name, storage in output_arguments.flatten.items():
            block.add_output_port(
                self.ctx.gen_var_name(name, block, 'gen'), storage)

    def process(self, node: ast.FunctionDef, parent=None):
        if not isinstance(node, ast.FunctionDef):
            raise TypeError('Expect as.FunctionDef')

        self.parse_func_decl(node)

        # pprint(node)
        for stmt in node.body:
            self.visit(stmt)


class ClassDefVisitor(AstVisitor):
    """
    Defining a user-defined type (like struct in C) by inheriting from
    `typing.NamedTuple` is the only case in which the keyword `class`
    is allowed to use.

    Inheriting from typing.NamedTuple makes it possible to use two ways to
    access fields of a user-defined type:
        1. indexing operation []
        2. member access operation .

    Using NamedTuple will require name resolver to resolve alias when parsing IR
    representation from the Python AST.
    """

    def __init__(self, ctx):
        super(ClassDefVisitor, self).__init__(ctx)

        if len(self.ctx):
            raise ParseError(('Function definition cannot be inside '
                              'a user function.'))

    def process(self, node: ast.ClassDef, parent=None):
        if not isinstance(node, ast.ClassDef):
            raise TypeError('Expect as.ClassDef')

        assert len(node.bases) == 1
        if node.bases[0].id != 'NamedTuple':
            raise UnsupportedType(("Only inheriting from NamedTuple is "
                                   "supported to define a user-defined type."))

        type_name = node.name
        if type_name in self.ctx.global_type_def:
            return

        parent = StorageInfoTree(type_name, None, None)
        for attr in node.body:
            if not isinstance(attr, ast.AnnAssign):
                # FIXME(ying): once inheriting from `typing.NamedTuple`,
                # each attribute will be binded to a type, attributes in
                # `node.body` will all have a type of ast.AnnAssign.
                # not sure in which condition this branch will be hit. keep
                # this check to defense error if there is.
                raise AnnotationError('Type annotation is required.')

            attr_name = attr.target.id
            child = StorageInfoTree('@{}'.format(attr_name), None, parent)
            if isinstance(attr.annotation, ast.Subscript):
                self.parse_type_annotation(attr.annotation, child)
            elif isinstance(attr.annotation, ast.Name):
                attr_type_name = attr.annotation.id
                if not (attr_type_name in self.ctx.global_type_def):
                    raise UnsupportedType()

                assert self.ctx.global_type_def[attr_type_name].parent is None

                self._add_as_child(child,
                                   self.ctx.global_type_def[attr_type_name])
            else:
                raise UnsupportedConstruct()
        self.ctx.global_type_def[type_name] = parent


class ReturnVisitor(AstVisitor):
    def __init__(self, ctx):
        super(ReturnVisitor, self).__init__(ctx)

    def names_in_source(self, value_node) -> List[Union[str, None]]:
        """
        Given the returned statement, return the name of returned value
        in the source code.

        If the return statement does not return a variable, but returns the
        evaluation of another statement, `None` will be returned.
        """

        if isinstance(value_node, ast.Name):
            return value_node.id
        elif isinstance(value_node, ast.Tuple):
            names = []
            for item in value_node.elts:
                names.append(__names_in_source(item))
            return names
        else:
            raise ParseError(('return statement should return variables '
                              f'or tuple of variables, but got {value_node}'))

    def get_returned_vars(self, value):
        def _get_var(value):
            if not isinstance(value, ast.Name):
                raise UnsupportedConstruct
            return value.id

        if isinstance(value, ast.Name):
            return [value.id]
        elif isinstance(value, ast.Tuple):
            return [_get_var(n) for n in value.elts]
        else:
            raise UnsupportedConstruct()

    def process(self, node: ast.Return, parent=None) -> NodeBase:
        if not isinstance(node, ast.Return):
            raise TypeError('Expect ast.Return')
        assert len(self.ctx.peek().partially_parsed_blocks) == 1

        rv_vars = self.get_returned_vars(node.value)
        block = self.ctx.peek().partially_parsed_blocks.pop()
        assert len(block.output_ports) == len(rv_vars)

        for rv_var, block_out in zip(rv_vars, block.output_ports):
            opnodes, ports = self.ctx.search_var_generation(rv_var)
            assert len(opnodes) and len(ports)

            for node, port in zip(opnodes, ports):
                block.add_output_node(node, (port, block_out))

        return block


class AssignVisitor(AstVisitor):
    """
    There are ONLY two cases of using assignment (behaviors of the assignment):
    1. evaluate a function call to create a new value.
    2. unpack tuple element.
    """

    def __init__(self, ctx):
        super(AssignVisitor, self).__init__(ctx)

    def _check_lhs(self, target):
        """
        Left-hand side of an assignment could only be one of:
            1. ast.Name, indicating a variable name;
            2. ast.Tuple with each elements being ast.Name
        """
        if isinstance(target, ast.Name):
            return False
        elif isinstance(target, ast.Tuple):
            for var in target.elts:
                self._check_lhs(var)
            return True
        else:
            raise UnsupportedConstruct(
                ('Left-hand side of an assignment statement must be '
                 f'varialbe(s), but got {type(target).__name__}.'))

    def _check_rhs(self, value):
        """
        Right-hand side of an assignment could only be one of:
            1. ast.Call, ast.BinOp, ast.UnaryOp which indicate primitive
               function calls.
            2. ast.Name, there is only situation for this case: unpacking
               tuple element.
            3. ast.Tuple with each element being ast.Call or ast.BinOp
        """

        if isinstance(value, ast.Call):
            return
        elif isinstance(value, ast.BinOp) or isinstance(value, ast.UnaryOp):
            """
            statement like z = x @ y hit hits branch.
            """
            return
        elif isinstance(value, ast.Name):
            # unpacking Tuple elements hitis this branch.
            return
        elif isinstance(value, ast.Subscript):
            return
        else:
            raise UnsupportedConstruct(
                ('Right-hand side of an assignment statement must be a '
                 f'primitive function call, but got {type(value).__name__}.'))

    def parse_unpack_tuple_elements(self, lhs: List[str], rhs_var: str):
        nodes, ports = self.ctx.search_var_generation(rhs_var)
        assert nodes and ports
        assert len(nodes) == len(lhs)
        symbol_table = self.ctx.peek().symbols
        block = self.ctx.peek().partially_parsed_blocks[-1]

        for alias, name in zip(lhs, ports):
            record = NameRecord(
                gen_name=name,
                node=None,
                block=block.name,
                level=block.depth,
                var_type='alias')
            symbol_table.insert(alias, record)
        return

    def process(self, node: ast.Assign, parent=None) -> NodeBase:
        """
        If the rhs is ast.Call/ast.BinOp/ast.UnaryOp:
            1. recursively parse the rhs.
            2. parse the lhs and add the output port.
        """
        if not isinstance(node, ast.Assign):
            raise TypeError('Expect ast.Assign')

        if len(node.targets) > 1:
            raise UnsupportedConstruct('Multiple assignment is not supported.')

        self._check_lhs(node.targets[0])

        self._check_rhs(node.value)

        if (isinstance(node.targets[0], ast.Tuple)
                and isinstance(node.value, ast.Name)):
            # assignment is used to unpack tuple element.
            self.parse_unpack_tuple_elements(
                [n.id for n in node.targets[0].elts], node.value.id)
            return None

        # recursively parse into the right-hand side, then left-hand side.
        opnode = self.visit(node.value)

        lhs = node.targets[0]
        lhs_name = [lhs.id] if isinstance(
            lhs, ast.Name) else [n.id for n in lhs.elts]

        opnode.output_ports.clear()
        if isinstance(opnode, ParallelNode):
            old_bodyout_names = list(opnode.bodyout.keys())
            opnode.bodyout.clear()

            if len(lhs_name) != len(old_bodyout_names):
                raise NotImplementedError()

            for n1, n2 in zip(lhs_name, old_bodyout_names):
                n = self.ctx.gen_var_name(n1, opnode, 'gen')
                opnode.add_output_port(n, None)
                new_bodyout_name = opnode.get_bodyout_port_name(n)
                opnode.update_edge_tip('out', EdgeEnd(opnode.name, n2),
                                       EdgeEnd(opnode.name, new_bodyout_name))

        elif isinstance(opnode, OperationNode):
            for name in lhs_name:
                opnode.add_output_port(
                    self.ctx.gen_var_name(name, opnode, 'gen'), None)

        else:
            raise ParseError(('Error node type. '
                              'Expected ParallelNode or OperationNode, '
                              f'got {type(opnode).__name__}.'))

        if isinstance(opnode, ParallelNode) and len(opnode.out_edges) == 0:
            """A parallel pattern like below will hit this branch:

            ys = ops.map(lambda x: ops.tanh(x), xs)
            """
            output_node = opnode.search_output_node()
            assert len(opnode.output_ports) == len(output_node.output_ports)
            for block_out, output_port in zip(opnode.output_ports,
                                              output_node.output_ports):
                opnode.add_output_node(output_node, (output_port, block_out))

        return opnode


class CallVisitor(AstVisitor):
    """A Call in kaleido will ONLY be one of the three cases:

    1. a parallel pattern (a ParallelNode in the IR program), the first argument
       of which is a user-defined function (a BlockNode in the IR program).
    2. a primitive tensor operation (an OperationNode in the IR program);
    3. a user-defined function (a BlockNode in the IR program, which in turn
       is made up of 1 and 2);
    """

    def __init__(self, ctx):
        super(CallVisitor, self).__init__(ctx)

    def create_parallel_node(self, opcode, lhs_vars: List[str],
                             rhs_vars: List[str]) -> ParallelNode:
        input_ports = OrderedDict()
        for var in rhs_vars:
            input_ports[var] = None

        output_ports = OrderedDict()
        for var in lhs_vars:
            output_ports[var] = None

        if opcode in kaleido.parser.context._APPLY_TO_EACH:
            return ApplyToEach(
                self.ctx.gen_op_name(),
                opcode,
                input_ports=input_ports,
                output_ports=output_ports)
        elif opcode in kaleido.parser.context._AGGREGATE:
            # FIXME(ying): by default `scan` and `fold` is interpreted as
            # left associative.
            if opcode == 'scan' or opcode == 'fold':
                opcode = '{}l'.format(opcode)

            return Aggregate(
                self.ctx.gen_op_name(),
                opcode,
                input_ports=input_ports,
                output_ports=output_ports)
        else:
            raise NotImplementedError()

    def create_tensor_opreation_node(self, opcode: str,
                                     args: List[ast.AST]) -> NodeBase:
        # a primitive tensor operation
        op = (registers.tensor_primitives[opcode]
              if opcode in registers.tensor_primitives else
              registers.access[opcode])
        node = op(self.ctx.gen_op_name())

        has_starred = sum([isinstance(arg, ast.Starred) for arg in args])
        if has_starred > 1:
            raise NotImplementedError()

        n = 0  # counts how many real parameters a starred parameter stands for
        if has_starred == 1:
            i = 0
            while not isinstance(args[i], ast.Starred):
                i += 1

            var = self.visit(args[i])
            _, ports = self.ctx.search_var_generation(var)
            n = len(ports)

        if has_starred:
            assert len(args) - 1 + n == node.arity
        else:
            if node.arity != -1:  # varadic function
                assert node.arity == len(args)

        for arg in args:
            if isinstance(arg, ast.Name) or isinstance(arg, ast.Attribute):
                name = self.visit(arg)
                node.add_input_port(self.ctx.gen_var_name(name, node, 'use'))
            elif isinstance(arg, ast.Starred):
                name = self.visit(arg)
                for _ in range(n):
                    node.add_input_port(
                        self.ctx.gen_var_name(name, node, 'use'))
            elif isinstance(arg, ast.List):
                parsed_args = self.visit(arg)
                for arg in parsed_args:
                    if not isinstance(arg, str):
                        raise NotImplementedError()
                if node.arity == -1:
                    node.arity = len(parsed_args)
                for arg_name in parsed_args:
                    node.add_input_port(
                        self.ctx.gen_var_name(arg_name, node, 'use'))
            else:
                node.add_input_port(self.ctx.gen_var_name(None, node, 'use'))
        return node

    def inline_udf(self, func_name,
                   func_args: List[ast.AST]) -> List[NodeBase]:
        """
        Args:
            func_name, str, the name of UDF.
            func_args, List[ast.AST],
        """
        cur_frame = self.ctx.peek()

        # `block` is the current BlockNode on top of the stack that is
        # partially parsed. Particularly, since a UDF is ONLY enclosed
        # inside parallel patterns or another UDF, `block` represents
        # the parent scope that contains `node`.
        block = cur_frame.partially_parsed_blocks[-1]
        node = self.ctx.frames[func_name].ir_block

        # inline UDF in `block`
        # Add all nodes and all internal edges in the parsed UDF into the
        # current BlockNode.
        for _, child_node in node.nodes.items():
            del child_node.parents[node.name]

            block.add_node(child_node)
            if isinstance(child_node, ParallelNode):
                child_node.increase_depth()
        for internal_edge in node.edges:
            block.edges[internal_edge] = node.edges[internal_edge]

        # resolve name alias between function arguments and their formal
        # parameters.
        # connect dataflow edges

        self.resolve_formal_parameter_name_alias(func_args, node)

        rv_nodes = []
        out_edges = node.out_edges
        rv_nodes = [block.nodes[tail.node] for tail in out_edges]

        return rv_nodes

    def create_ir_node(self, func_name: str,
                       func_args: List[ast.AST]) -> NodeBase:
        """
        1. create a BlockNode or OperationNode from ast.Call's func field.
        2. push the IR node into the stack that stores the partially parsed IR
           node.
        3. add the created IR node into the BlockNode which is the parent IR
           node of this newly created IR node.

        Returns:
            If `func_name` is a user-defined function, it should be parsed
            already. `create_ir_node` will return the parsed BlockNode. Raise
            error if BlockNode that correspond to the user-defined function
            is not found.
        """

        node = None
        block = self.ctx.peek().partially_parsed_blocks[-1]

        if func_name in kaleido.parser.context._PARALLEL_PATTERNS:
            # a parallel pattern
            node = self.create_parallel_node(func_name, [], [])

            if len(func_args) != 2:
                raise ParseError(
                    ('A parallel pattern in kaleido exactly accepts '
                     'two arguments(the first one is a user-defined '
                     'and the second one is the access pattern), '
                     f'but got {len(func_args)}.'))

        elif (func_name in registers.tensor_primitives
              or func_name in registers.access):
            node = self.create_tensor_opreation_node(func_name, func_args)
        else:
            raise UnknownPrimitiveOps(f'Unknown function {func_name}.')

        self.ctx.peek().partially_parsed_nodes.append(node)
        block.add_node(node)
        return node

    def check_is_supported_func(self, func_name: str, args):
        """
        There are only three kinds of function:
        1. parallel pattern: a ParallelNode.
        2. primitive operation: an OperationNode.
        3. user-defined function made up from the above two: a BlockNode.
        """

        if func_name in kaleido.parser.context._PARALLEL_PATTERNS:
            return
        elif func_name in registers.tensor_primitives:
            return
        elif func_name in self.ctx.frames or func_name in registers.access:
            # user-defined function that has already parsed or access pattern
            return
        else:
            raise UnknownPrimitiveOps(f'Unknown function: {func_name}.')

    def parse_keyword_arguments(self, keywords: List[ast.AST]):
        """parse function's keyword arguments.

        Note:
        1. Aggregate pattern's keyword 'initializer' could be a computation
            graph, thus requires a recursive parsing.
        1. Except 'initializer', current implementation interpretes all other
           keyword as attributes of a primitive operations. This is limited,
           and requres a further analysis and check.
        """

        init_node = []
        ir_node = self.ctx.peek().partially_parsed_nodes[-1]
        for keyword in keywords:
            if keyword.arg == 'initializer':
                if isinstance(keyword.value, ast.Name) or isinstance(
                        keyword.value, ast.Call):
                    init_node.append(self.visit(keyword.value))
                elif isinstance(keyword.value, ast.Tuple):
                    for keyword in keyword.value.elts:
                        init_node.append(self.visit(keyword))
                else:
                    raise NotImplementedError()

                continue

            if not (isinstance(keyword.value, ast.List)
                    or isinstance(keyword.value, ast.Tuple)
                    or isinstance(keyword.value, ast.Num)
                    or isinstance(keyword.value, ast.Str)):
                raise UnsupportedConstruct(
                    (f'Unsupported type: {type(keyword.value).__name__} '
                     f'for keyword argument {keyword.arg}.'))

            ir_node.attributes[keyword.arg] = self.visit(keyword.value)
        return init_node

    def parse_initializer(self, cur_op: NodeBase, func_name: str,
                          keywords) -> int:
        init_node = self.parse_keyword_arguments(keywords)

        block_node = self.ctx.peek().partially_parsed_blocks[-1]
        if not init_node:
            return

        assert func_name in kaleido.parser.context._AGGREGATE

        for i, init in enumerate(init_node):
            if isinstance(init, str):
                # initializer is a variable that has already been defined.
                target_nodes, target_ports = self.ctx.search_var_generation(
                    init)
                assert len(target_nodes) and len(target_ports)

                for j, (target_node, target_port) in enumerate(
                        zip(target_nodes, target_ports)):
                    init_port = self.ctx.gen_var_name(
                        '{}@state_init.{}.{}'.format(cur_op.name, i, j),
                        cur_op, 'use')
                    cur_op.add_state_init_port(init_port)

                    block_node.add_edge(
                        tail=(target_node, target_port),
                        tip=(cur_op, init_port),
                        edge_type='in'
                        if target_node.name == block_node.name else None)

                    state_port = self.ctx.gen_var_name(cur_op.name, cur_op,
                                                       'def')
                    cur_op.add_state_port(state_port)
            elif isinstance(init, NodeBase):
                if init.arity:
                    # this branch is hit if the initializer is not a constant node
                    raise NotImplementedError()
                block_node.add_node(init)

                assert cur_op.opcode in kaleido.parser.context._AGGREGATE

                port = self.ctx.gen_var_name(
                    '{}@state_init.{}'.format(cur_op.name, i), cur_op, 'use')
                cur_op.add_state_init_port(port, None)

                # data flows from the output port of the `initializer` to
                # `_state_init` port of the `cur_op` which is an Aggregate pattern.
                block_node.add_edge(
                    tail=(init, next(reversed(init.output_ports))),
                    tip=(cur_op, port))

                state_port = self.ctx.gen_var_name(
                    '{}@state.{}'.format(cur_op.name, i), cur_op, 'def')
                cur_op.add_state_port(state_port)
            else:
                raise ParseError()

    def add_inport_to_parallel_node(self, block: BlockNode, var_name: str):
        def _find_parent_block(block):
            for b in self.ctx.peek().partially_parsed_blocks:
                if b.name == block.name or block.name in b.nodes:
                    return b

        if not isinstance(block, BlockNode):
            raise TypeError((f'Expect BlockNode, got {block.name}: '
                             '{type(block).__name__}.'))

        input_port_name = self.ctx.gen_var_name(var_name, block, 'use')
        block.add_input_port(input_port_name)
        block.arity += 1

        target_node, target_port = self.ctx.search_var_generation(var_name)
        assert len(target_port) == 1 and len(target_node) == 1
        target_node, target_port = target_node[0], target_port[0]

        if not target_port:
            raise ParseError(f'Unknown variable: {var_name}')

        parent_block = _find_parent_block(block)

        if parent_block.name == target_node.name:
            edge_type = None
            if target_port.endswith(
                    'bodyin') or target_port in target_node.input_ports:
                edge_type = 'in'
            target_node.add_edge(
                tail=(target_node, target_port),
                tip=(block, input_port_name),
                edge_type=edge_type)
        else:
            # FIXME(ying): check whether this assert fail in the
            # stacked LSTM example.
            # assert parent_block.name == _find_parent_block(target_node).name

            if target_port.endswith(
                    'bodyin') or target_port in target_node.input_ports:
                edge_type = 'in'
            parent_block.add_edge(
                tail=(target_node, target_port), tip=(block, input_port_name))

    def parse_parallel_pattern_input(self, parallel_node: ParallelNode,
                                     xs: ast.AST):
        if isinstance(xs, ast.Name):
            self.add_inport_to_parallel_node(parallel_node, xs.id)
        elif isinstance(xs, ast.Call) and xs.func.attr == 'zip':
            for arg in xs.args:
                if not (isinstance(arg, ast.Name)
                        or isinstance(arg, ast.Attribute)):
                    raise NotImplementedError()
                self.add_inport_to_parallel_node(parallel_node,
                                                 self.visit(arg))
        else:
            input_node = self.visit(xs)

            parent_block = None
            for b in self.ctx.peek().partially_parsed_blocks:
                if (b.name == parallel_node.name
                        or parallel_node.name in b.nodes):
                    parent_block = b
                    break

            assert parent_block
            for outport in input_node.output_ports:
                input_port_name = self.ctx.gen_var_name(
                    outport, parallel_node, 'use')
                parallel_node.add_input_port(input_port_name)

                parent_block.add_edge(
                    tail=(input_node, outport),
                    tip=(parallel_node, input_port_name))

    def get_parsed_formal_param(
            self, parsed_udf: BlockNode) -> Tuple[List[NodeBase], List[str]]:
        """
        Returns the compiler generated names for a UDF's formal parameters.

        Example:
            def udf(x, y, z):
                v1 = op1(x)
                v2 = op2(y, z)
                v3 = op3(v1, v2)
                return v3

        in the parsed codes, a UDF is represented by a BlockNode, called it `BN`,
        and `x`, `y`, `z` are input ports of `BN`.

        In the above code example, dataflow information for udf's formal
        parameters `x`, `y`, `z` are:

            the 1-st input port of BN (define x) -->
                the 1-st input port x' of op1 (consume x)
            the 2-nd input port of BN (define y) -->
                the 1-st input port of y' op2 (consume y)
            the 3-rd input port of BN (define z) -->
                the 2-nd input port z' of op2 (z)

        This functions returns the compiler generated names for ports:
        x', y', and z'(including node name and port name).
        """
        assert isinstance(parsed_udf, BlockNode)

        formal_params_names = []
        formal_params_nodes = []
        in_edges = parsed_udf.in_edges

        for i, tail_port in enumerate(parsed_udf.bodyin.keys()):
            tail = EdgeEnd(node=parsed_udf.name, port=tail_port)
            if not tail in in_edges:
                # the `i-th` input argument is consumed by a ParallelNode
                # inside the UDF which is not the lead node of the UDF,
                # thus does not appear as a tail port of any in_edges.
                formal_params_nodes.append(parsed_udf)
                formal_params_names.append(tail_port)
            else:
                for tip in in_edges[tail]:
                    # the tail of an `in_edge` always points to an input port of
                    # a node inside the UDF.
                    formal_params_nodes.append(parsed_udf.nodes[tip.node])
                    formal_params_names.append(tip.port)

        return formal_params_nodes, formal_params_names

    def resolve_formal_parameter_name_alias(self, args: List[ast.AST],
                                            func_node: NodeBase):
        """
        See the example below:

        Example:
            def udf(x, y, z):
                v1 = op_1(x)
                v2 = op_2(y, z)
                v3 = op_3(v1, v2)
                return v3

            a = op1(...)
            b = op2(...)
            c = op3(...)

            d = udf(a, b, c)

        in the parsed codes, a UDF is represented by a BlockNode, called it `BN`,
        and `x`, `y`, `z` are input ports of `BN`.

        NOTE, caller of this function will INLINE the UDF, and this function
        create dataflow edges:

             output port of op1 (produce a) -->
                the 1-st input port of op_1 (consume a)
             output port of op2 (produce b) -->
                the 1-st input port of op_2 (consume b)
             output port of op3 (produce c) -->
                the 2-nd input port of op_2 (consume c)

        Args:
            args, List[ast.AST], variable names of arguments in the source
                                 codes that are passed to evaluate the function.
            func_node, BlockNode, the parsed UDF.
        """

        def _update_tail(block_node: BlockNode, port: str):
            tip_node = None
            tip_port = None
            for node_name, node in block_node.nodes.items():
                tail = EdgeEnd(block_node.name, port)
                if isinstance(node, ParallelNode):
                    if tail in node.in_edges:
                        tip = node.in_edges[tail]

                        if len(tip) > 1:
                            raise ParseError()

                        tip = tip[0]
                        tip_node = node.nodes[tip.node]
                        tip_port = tip.port

                        del node.in_edges[tail]

                        node.add_edge(
                            (alias_nodes[i], alias_ports[i]),
                            (tip_node, tip_port),
                            edge_type='in',
                            check_edge=False)
                    elif node.opcode == 'block':
                        raise ParseError(('BlockNode only appears '
                                          'at the outmost depth.'))
            assert tip_node and tip_port

        is_starred = [isinstance(arg, ast.Starred) for arg in args]
        if sum(is_starred) > 1:
            raise NotImplementedError(("More than 1 starred arguments "
                                       "is not supported yet."))

        fp_nodes, fp_gen_names = self.get_parsed_formal_param(func_node)

        cur_frame = self.ctx.peek()
        symbol_table = cur_frame.symbols
        block = cur_frame.partially_parsed_blocks[-1]  # get parent scope

        idx = 0
        for pos, arg in enumerate(args):
            var_name = self.visit(arg)
            if not isinstance(var_name, str):
                raise ParseError(
                    f"Exprected string, got {type(name).__name__}.")
            if var_name not in symbol_table:
                # `args` are actual parameters name of the actual parameter
                # in the source code to evaluate `func_node`, thus they should
                # be in the symbol table already.
                raise ValueError(f'Unknown variable {name}.')

            if isinstance(arg, ast.Starred):
                assert len(fp_gen_names) > len(args)
                # how many variables the starred parameter stands for
                num = len(fp_gen_names) - len(args) + 1

                alias_nodes, alias_ports = self.ctx.search_var_generation(
                    var_name)
                if num != len(alias_ports):
                    raise ValueError(('Inconsistent number of tuple elements:'
                                      f' {num} vs. {len(alias_nodes)}.'))
                for i in range(num):
                    if fp_nodes[idx].opcode == 'block':
                        _update_tail(fp_nodes[idx], fp_gen_names[idx])
                    else:
                        node = block.nodes[fp_nodes[idx].name]
                        gen_name = fp_gen_names[idx]

                        edge_type = None
                        if (alias_ports[i].endswith('bodyin')
                                or alias_ports[i] in block.input_ports):
                            edge_type = 'in'

                        # FIXME(ying): refactor the logic of symbol table search.
                        # the current implmentation is error-prone.
                        block.add_edge((alias_nodes[num - i - 1],
                                        alias_ports[num - i - 1]),
                                       (node, gen_name), edge_type)
                    idx += 1

            elif (isinstance(arg, ast.Name) or isinstance(arg, ast.Attribute)):
                if fp_nodes[idx].opcode == 'block':
                    _update_tail(fp_nodes[idx], fp_gen_names[idx])
                else:
                    node = block.nodes[fp_nodes[idx].name]
                    gen_name = fp_gen_names[idx]

                    alias_nodes, alias_ports = self.ctx.search_var_generation(
                        var_name)
                    assert len(alias_nodes) == len(alias_ports) == 1

                    # if alias_nodes[0].opcode == 'block':
                    block.add_edge(
                        (alias_nodes[0], alias_ports[0]), (node, gen_name),
                        edge_type='in',
                        check_edge=(not alias_nodes[0].opcode == 'block'))

                idx += 1
            else:
                raise NotImplementedError()

    def create_dataflow_edge_for_str_arg(self, arg: str, opnode: NodeBase,
                                         block: BlockNode):
        """
        Create dataflow edge flowing as:
            arg (indicating tail) --> the input port of opnode (indicating tip)
                                      that consumes `arg`.

        Args:
            arg, `str`, the `arg` is a variable which has already been defined
                        (a `gen` record exists in the symbol table).
            opnode, NodeBase,
            block, BlockNode, the scope where the created dataflow edge exist.
        """
        tail_nodes, tail_ports = self.ctx.search_var_generation(arg)
        tip_ports = self.ctx.search_var_useage(arg, opnode.name, opnode)
        assert len(tail_nodes) == len(tail_ports) == len(tip_ports)

        for tail_node, tail_port, tip_port in zip(tail_nodes, tail_ports,
                                                  tip_ports):
            tip = (opnode, tip_port)
            edge_type = None
            if tail_port.endswith('bodyin') or tail_port in block.input_ports:
                edge_type = 'in'

            block.add_edge((tail_node, tail_port), tip, edge_type)

    def create_dataflow_edges(self, opnode: NodeBase,
                              parsed_args: List[Union[str, NodeBase]]):
        """
        Add dataflow edges flowing from:
            parsed_args(tails) --> input ports of opnode (tips).

        Example:
            ...
            rv = f(x, y, z)
            ...

        Args:
            opnode, NodeBase, stands for the function to be evaluated.
                              In the example above, opnode is the IR Node for
                              function `f`, which is either an OperationNode,
                              or a ParallelNode.
            parsed_args, List[Union[str, NodeBase]], in the above example,
                         `parsed_args` are actual parameters `x`, `y`, `z`.
                         For a single `parsed_arg`, there are two cases:
                         1. has a type of NodeBase: stands for a parsed IR
                            node whose output port (restricted
                            to have exactly 1 output port) produces one actual
                            parameter.
                         2. has a type of plain string.
        """
        block = self.ctx.peek().partially_parsed_blocks[-1]
        assert isinstance(block, BlockNode)

        for i, arg in enumerate(parsed_args):
            if isinstance(arg, str):
                # the `arg` is a variable which has already been defined.
                self.create_dataflow_edge_for_str_arg(arg, opnode, block)
            elif isinstance(arg, NodeBase):
                assert len(arg.output_ports) == 1
                tail = (arg, next(reversed(arg.output_ports)))
                tip = (opnode, opnode.get_input_port_name(i))
                block.add_edge(tail, tip)
            elif isinstance(arg, List):
                for elem in arg:
                    if not isinstance(elem, str):
                        raise UnsupportedConstruct()
                for elem in arg:
                    self.create_dataflow_edge_for_str_arg(elem, node, block)
            else:
                raise UnsupportedConstruct()

        if opnode.name == self.ctx.peek().partially_parsed_blocks[-1].name:
            self.ctx.peek().partially_parsed_blocks.pop()

    def process(self, node: ast.Call, parent=None) -> BlockNode:
        """Parsing ast.Call is at the core in kaleido.

        In general, the process goes as:
        1. step 1. parse function name and create an IR node. An IR node can only
           be either a BlockNode(including ParallelNode) or an OperationNode
           (a primitive Tensor operation).

           After creation:
           1). if the IR node is a UDF (a parsed BlockNode), it is "inlined".
           2). if the IR node is not a UDF, then such an IR node is
               just partially parsed (only function name is parsed, recursive
               parsing of its arguments is required) and is pushed into a stack
               that stores partially parsed IR nodes while traversing the
               Python AST.

        2. step 2. parse Aggregate pattern's initializer if it is given.
        3. step 3. if `node's first argument is ast.Lambda, indicating a
           parallel pattern, parse node's second argument (a FactorTensor to
           iterate over), because it lives in the same scope as `node`.
           then goest to 4.
        4. step 4. add the IR node created in step 1 into stack.
        5. step 5. if the `node` is a parallel pattern:
            1) parse access pattern FIRST
            2) parse lambda body SECOND
        6. step 6. recursively parse function arguments. An argument can ONLY be
            one of the following cases:
            1). ast.Name: a variable.
            2). ast.BinOp/ast.UnaryOp (Python's built-in arithmetic operator,
                like +. -. @ which are overloaded into tensor operations.)
            3). ast.Lambda: is only allowed for parallel patterns.
        7. create data flow edges.
        """
        if not isinstance(node, ast.Call):
            raise TypeError('Expect ast.Call')

        func_name = self.parse_func_name(node.func)
        self.check_is_supported_func(func_name, node.args)

        if func_name in self.ctx.frames:
            # a user-defined function that has been parsed already
            return self.inline_udf(func_name, node.args)

        cur_op = self.create_ir_node(func_name, node.args)
        self.parse_initializer(cur_op, func_name, node.keywords)

        is_parallel_pattern = (len(node.args)
                               and isinstance(node.args[0], ast.Lambda))
        if is_parallel_pattern:
            assert len(node.args) == 2
            self.parse_parallel_pattern_input(cur_op, node.args[1])

        if isinstance(cur_op, BlockNode):
            """
            If the newly created IR node `cur_op` is a BlockNode, and it is
            not a user-defined function that has already completed parsing,
            push `cur_op` to the stack that stores partially parsed BlockNodes.
            """
            self.ctx.peek().partially_parsed_blocks.append(cur_op)

        parsed_args = []
        if is_parallel_pattern:
            self.visit(node.args[0])  # call ast.Lambda's visitor.
        else:
            # parse arguments of an OperationNode
            parsed_args = [self.visit(arg) for arg in node.args]

        fully_parsed = self.ctx.peek().partially_parsed_nodes.pop()

        if not isinstance(fully_parsed, ParallelNode):
            fully_parsed.add_output_port(
                self.ctx.gen_var_name(None, fully_parsed, 'gen'))

        self.create_dataflow_edges(fully_parsed, parsed_args)

        if isinstance(fully_parsed, ParallelNode):
            self.ctx.clean_symbols(fully_parsed.name)
        return fully_parsed


class LambdaVisitor(AstVisitor):
    """Lambda is ONLY allowed to be the first argument of a parallel pattern."""

    def __init__(self, ctx):
        super(LambdaVisitor, self).__init__(ctx)

    def process(self, node: ast.Lambda, parent=None) -> BlockNode:
        """
        In kaleido's program ast.Lambda is only allowed to be used for the first
        argument of a parallel pattern (a ParallelNode in the IR
        representation).

        Any ParallelNode has at least one required argument: the iterative
        FractalTensor as its input; call it `xs`. At the time when recursive
        parsing calls ast.Lambda's visitor, ONLY the input for `xs` is created,
        except that the ParallelNode is an empty template.
        """
        if not isinstance(node, ast.Lambda):
            raise TypeError('Expect ast.Lambda')

        block = self.ctx.peek().partially_parsed_blocks[-1]
        if not isinstance(block, ParallelNode):
            raise ParseError(
                f'Expected ParallelNode, got {type(block).__name__}')

        arg_names = [arg.arg for arg in node.args.args]
        if not (len(arg_names) == 1 or len(arg_names) == 2):
            raise ParseError(('Lambda is only allowed to have 1 argument '
                              '(ApplyToEach pattern) '
                              'or 2 argument (Aggregate pattern).'))

        in_ids = 0 if len(arg_names) == 1 else 1

        symbol_table = self.ctx.peek().symbols

        if len(arg_names) == 2:
            if len(block._state):
                for state_port in block._state:
                    record = NameRecord(
                        gen_name=state_port,
                        node=block,
                        block=block.name,
                        level=block.depth,
                        var_type='def')
                    symbol_table.insert(arg_names[0], record)
            else:
                raise NotImplementedError(
                    ('An explicit initializer should be given to '
                     'the Aggregate pattern.'))

        for bodyin in block.bodyin:
            record = NameRecord(
                gen_name=bodyin,
                node=None,
                block=block.name,
                level=block.depth,
                var_type='alias')
            symbol_table.insert(arg_names[in_ids], record)
            record = NameRecord(
                gen_name=bodyin,
                node=block,
                block=block.name,
                level=block.depth,
                var_type='def')
            symbol_table.insert(bodyin, record)

        # A BlockNode or an OperationNode is returned.
        body_node = self.visit(node.body)

        num_out = 0
        if isinstance(body_node, List):
            # NOTE: in the current implementations, when lambda body is a UDF,
            # this branchs is hit.
            assert len(block.output_ports) == 0
            num_out = len(body_node)
            for i in range(num_out):
                block.add_output_port(
                    self.ctx.gen_var_name(None, block, 'gen'))

                if len(body_node[i].output_ports) > 1:
                    if isinstance(body_node[i], ParallelNode):
                        logging.warning(
                            ('\nLambda body is a parallel function '
                             'that returns more than 1 values, '
                             'but current implementation only returns '
                             'the first returned value to the parent scope.'))
                    else:
                        raise NotImplementedError(
                            ("OperationNode that has "
                             "more than 1 outputs is not supported yet."))

                tail_node = body_node[i]
                tail_port = list(body_node[i].output_ports.keys())[0]

                block.add_output_node(
                    tail_node, (tail_port, block.get_output_port_name(i)))
                return body_node
        elif (isinstance(body_node, ParallelNode)
              or isinstance(body_node, OperationNode)):
            num_out = len(body_node.output_ports)
            assert num_out and len(block.output_ports) == 0
            for i in range(num_out):
                block.add_output_port(
                    self.ctx.gen_var_name(None, block, 'gen'))

            if len(block.out_edges) == 0:
                out_node = block.search_output_node()

                for port, block_out in zip(out_node.output_ports,
                                           block.output_ports):
                    block.add_output_node(out_node, (port, block_out))
            return body_node
        else:
            raise NotImplementedError('Lambda body should be an expression.')


class BinOpVisitor(AstVisitor):
    def __init__(self, ctx):
        super(BinOpVisitor, self).__init__(ctx)

    def parse_operand(self, cur_op: NodeBase, operand: ast.AST):
        block = self.ctx.peek().partially_parsed_blocks[-1]

        if isinstance(operand, ast.Name):
            op = self.ctx.peek().partially_parsed_nodes[-1]

            var = operand.id
            port_name = self.ctx.gen_var_name(
                name=None, node=op, var_type='use')
            op.add_input_port(port_name)

            tail_node, tail_port = self.ctx.search_var_generation(var)
            assert len(tail_node) == 1 and len(tail_port) == 1
            tail_node, tail_port = tail_node[0], tail_port[0]

            if tail_port in block.input_ports:
                block.add_input_node(op, (tail_port, port_name))
            else:
                block.add_edge(
                    tail=(tail_node, tail_port),
                    tip=(op, port_name),
                    edge_type='in' if tail_port.endswith('bodyin') else None)
        else:
            port_name = self.ctx.gen_var_name(
                name=None, node=cur_op, var_type='use')
            cur_op.add_input_port(port_name)

            operand = self.visit(operand)

            opnode = self.ctx.peek().partially_parsed_nodes[-1]
            if isinstance(operand, str):
                # a variable name
                nodes, ports = self.ctx.search_var_generation(operand)
                assert len(nodes) == len(ports) == 1

                block.add_edge(
                    tail=(nodes[0], ports[0]),
                    tip=(opnode, port_name),
                    edge_type='in' if ports[0].endswith('bodyin') else None)
            else:
                assert isinstance(operand, NodeBase)

                block.add_edge(
                    tail=(operand, next(reversed(operand.output_ports))),
                    tip=(opnode, port_name))

    def process(self, node: ast.BinOp, parent=None) -> NodeBase:
        if not isinstance(node, ast.BinOp):
            raise TypeError('Expect ast.BinOp')
        block = self.ctx.peek().partially_parsed_blocks[-1]

        op = registers.tensor_primitives[type(node.op).__name__]

        # create the OperationNode
        cur_op = op(self.ctx.gen_op_name())

        block.add_node(cur_op)
        self.ctx.peek().partially_parsed_nodes.append(cur_op)

        left_op = self.parse_operand(cur_op, node.left)
        right_op = self.parse_operand(cur_op, node.right)

        fully_parsed = self.ctx.peek().partially_parsed_nodes.pop()
        fully_parsed.add_output_port(
            self.ctx.gen_var_name(None, fully_parsed, 'gen'))

        return fully_parsed
