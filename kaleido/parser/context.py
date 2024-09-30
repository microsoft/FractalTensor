from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict
from typing import List
from typing import Tuple
from typing import Deque
from typing import NamedTuple
from collections.abc import Sequence
from collections import OrderedDict
from collections import deque
import warnings
from enum import Enum
from enum import unique
import kaleido
from kaleido.frontend.types import Storage
from kaleido.parser.ir_nodes import NodeBase
from kaleido.parser.ir_nodes import BlockNode

__all__ = [
    'Context',
]

_APPLY_TO_EACH = {'map'}
_AGGREGATE = {'reduce', 'fold', 'foldl', 'foldr', 'scan', 'scanr', 'scanl'}
_PARALLEL_PATTERNS = set.union(_APPLY_TO_EACH, _AGGREGATE)


class NameRecord(NamedTuple):
    gen_name: str  # compiler-generated port name
    node: NodeBase  # compiler-generated node name
    block: str  # parent block of the node
    level: int  # depth of the block, which should be less than `node.depth`
    var_type: str  # def, gen, use, or alias

    def __str__(self):
        print_str = '  {}\n'.format(self.gen_name)

        if self.node:
            print_str += '  - node name: {}\n'.format(self.node.name)
        else:
            print_str += '  - node name: None\n'

        print_str += '  - block name: {}\n'.format(self.block)
        print_str += '  - level: {}\n'.format(self.level)
        print_str += '  - type: {}\n'.format(self.var_type)
        return print_str

    __repr__ = __str__


class SymbolTable(object):
    def __init__(self):
        self._symbols = OrderedDict()

    def insert(self, name: str, record: NameRecord):
        if name in self._symbols:
            self._symbols[name].append(record)
        else:
            self._symbols[name] = [record]

    def keys(self):
        return self._symbols.keys()

    def __getitem__(self, key):
        try:
            return self._symbols[key]
        except KeyError as e:
            raise KeyError(f'variable {key} is not in the symbol table.')

    def __contains__(self, key):
        return key in self._symbols

    def __str__(self):
        print_str = ''
        for var, records in self._symbols.items():
            print_str += '{}:\n'.format(var)
            for record in records:
                print_str += '{}'.format(str(record))
        return print_str

    __repr__ = __str__

    def dump(self, filename='symbols.tmp'):
        import os
        with open(os.path.join(os.getcwd(), filename), 'w') as f:
            f.write(str(self))

    def clean_symbols(self, target_name: str):
        for name, records in list(self._symbols.items()):
            new_records = []
            for record in records:
                if record.block != target_name:
                    new_records.append(record)
                else:
                    # alias, def, and use is removed
                    if record.var_type == 'gen':
                        new_records.append(record)
            del self._symbols[name]
            if len(new_records):
                self._symbols[name] = new_records


class ContextFrame(object):
    """
    A ContextFrame stores the parsed IR program of a function decorated by
    `kaleido.function` decorator.

    Args:
        name, str: identifier of the context frame which is identical to the
              name of the outermost BlockNode.
        ir_block, BlockNode: each user function is parsed into a BlockNode and
                  BlockNodes can be nested. ir_block stores the reference to
                  the outermost BlockNode.
    """

    def __init__(self, name: str, ir_block: BlockNode):
        self._name = name

        if not isinstance(ir_block, BlockNode):
            raise TypeError('Expected BlockNode.')

        self._ir_block = ir_block

        # stack to store IR nodes that are not fully parsed yet.
        self.partially_parsed_blocks: Deque[BlockNode] = deque()
        self.partially_parsed_nodes: Deque[NodeBase] = deque()

        # names in the source code in the local scope of the function.
        self._symbols = SymbolTable()

    @property
    def name(self) -> str:
        return self._name

    @property
    def symbols(self) -> str:
        return self._symbols

    @property
    def ir_block(self) -> BlockNode:
        return self._ir_block

    def __str__(self):
        print_str = '\nContextFrame({})\n'.format(self._name)
        return print_str

    __repr__ = __str__


class Context(Sequence):
    """A stack to store partially parsed intermediate results."""

    def __init__(self):
        # user-defined types in the global scope.
        self.global_type_def: Dict = {}

        # each frames stores a parsed IR program for a function decorated
        # by the `function` decorator.
        self.frames: OrderedDict[str, ContextFrame] = OrderedDict()

        # FIXME(ying): a very buggy implementation to record whether the
        # function is "compiled". Fix it later.
        self._compiled = set()

        self.name_count = -1
        self.op_count = -1
        self.tmp_count = -1

    def __getitem__(self, i):
        return self.frames[list(self.frames.keys())[i]]

    def __len__(self) -> int:
        return len(self.frames)

    def reset_context(self):
        self.frames = OrderedDict()

        # count for compiler generated names.
        self.name_count = -1

        self.op_count = -1
        self.tmp_count = -1

        self.global_type_def = {}
        self._compiled = {}

    def gen_op_name(self):
        """Generate unique identifier for an operation node."""

        self.op_count += 1
        return f'%node{self.op_count}'

    def gen_var_name(self, name: str, node: NodeBase, var_type: str):
        """Generate unique identifier for a variable in the source code.

        Args:
            name, str, name in the source code.
            node, NodeBase, IR node that generates `name`. If None, `name` is a
                  name alias of another variable.

        Returns:
            the generated identifier is made up of three parts separated by '#':
                {function name}#{variable name in in the source code}#{version}

            For intermediate variables that do not have an explicit name in the
            source code, '%TMP' will be used as the second part.
        """

        assert self.peek()
        symbol_table = self.peek().symbols
        local_scope = self.peek().partially_parsed_blocks[-1]

        if node:
            assert isinstance(node, NodeBase)

        self.name_count += 1
        gen_name = '%{}'.format(self.name_count)

        record = NameRecord(
            gen_name=gen_name,
            node=node,
            block=local_scope.name,
            level=local_scope.depth,
            var_type=var_type)
        if name:
            if node is None:
                warnings.warn(
                    'operation that generates the varialbe is not given.')
            symbol_table.insert(name, record)
        else:
            if node is None:
                raise ValueError(
                    'the node that generates the variable must be given.')
            symbol_table.insert(gen_name, record)
        return gen_name

        self.name_count += 1
        return '{}#{}#{}'.format(func_name, '%TMP', self.name_count)

    def search_var_generation(self, var: str) -> List[Tuple[NodeBase, str]]:
        symbol_table = self.peek().symbols

        nodes = []
        ports = []
        if var in symbol_table:
            for occr in reversed(symbol_table[var]):
                if occr.var_type == 'use':
                    continue
                elif occr.var_type == 'alias':
                    node, port = self.search_var_generation(occr.gen_name)
                    assert len(node) == 1 and len(port) == 1
                    nodes.append(node[0])
                    ports.append(port[0])
                elif occr.var_type == 'gen':
                    return [occr.node], [occr.gen_name]
                elif occr.var_type == 'def':
                    nodes.append(occr.node)
                    ports.append(occr.gen_name)
                else:
                    raise ValueError(f'Unknown var_type: {occr.var_type}.')
            return nodes, ports
        else:
            # unpacking a Tuple reaches this branch
            for name in symbol_table.keys():
                if '@' in name:
                    if var in name:
                        occr = symbol_table[name]
                        assert len(occr) == 1
                        nodes.append(occr[0].node)
                        ports.append(name)
            return nodes, ports

    def search_var_useage(self,
                          var: str,
                          opname: str,
                          opnode: BlockNode = None) -> List[str]:
        """
        Args:
            var, str, variable name in the source codes.
            opname, str, target node name.
            opnode, BlockNode,

        Returns:
            port name of the parsed function and that port consumes
            the variable `var`. return `None` if the search fails.
        """
        if not isinstance(opname, str):
            raise ValueError(
                f'Expect name of an IR not, got {type(opname).__name__}')

        symbol_table = self.peek().symbols

        useages = []
        if var in symbol_table:
            for occurrence in reversed(symbol_table[var]):
                if (occurrence.var_type == 'use'
                        and occurrence.node.name == opname):
                    useages.append(occurrence.gen_name)
            return useages
        else:
            raise KeyError(f'Unknown name {var}.')

    def clean_symbols(self, block_name: str):
        self.peek()._symbols.clean_symbols(block_name)

    def gen_tmp_name(self):
        """Generate unique identifier for intermediate variable."""

        self.tmp_count += 1
        return f'%tmp{self.tmp_count}'

    def push(self, frame: ContextFrame):
        """
        Args:
            name, str:
            node, BlockNode:
        """
        if not isinstance(frame, ContextFrame):
            raise TypeError('Expected ContextFrame.')

        self.frames[frame.name] = frame

    def peek(self) -> ContextFrame:
        """Get or create a frame into the context.

        If the context is not empty, returns the top element of `self.frames`
        WITHOUT removing it.
        """

        return self.frames[next(reversed(self.frames))] if len(
            self.frames) else None
