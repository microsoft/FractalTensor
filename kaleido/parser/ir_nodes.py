from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from collections import OrderedDict
from abc import ABCMeta
from abc import abstractmethod
from typing import Union
from typing import List
from typing import Tuple
from typing import Dict
from typing import TypeVar

import kaleido
from kaleido.frontend.types import Storage
from kaleido.frontend.types import StorageInfoTree
from kaleido.frontend.types import TensorStorage
from kaleido.frontend.types import FractalTensorStorage
from kaleido.frontend.types import Number
from kaleido.parser.errors import ParseError

__all__ = [
    'ParallelNode',
    'BlockNode',
    'AccessNode',
    'BranchNode',
    'Port',
    'EdgeEnd',
    'ApplyToEach',
    'Aggregate',
]


class Port(object):
    """A port is a unique identifier(name) associated with Storage."""

    def __init__(self, name: str, storage: Storage = None):
        self._name = name
        self._storage = storage

    @property
    def name(self) -> str:
        return self._name

    @property
    def storage(self) -> Storage:
        return self._storage

    @storage.setter
    def storage(self, s: Storage):
        self._storage = s


PN = TypeVar('ParallelNode')
ON = TypeVar('OperationNode')


class NodeBase(object):
    """Base class for the IR Node."""

    arity = 0
    opcode = None

    def __init__(self,
                 name: str,
                 input_ports: Dict[str, Storage] = None,
                 output_ports: Dict[str, Storage] = None,
                 depth: int = 0,
                 parents: Dict[str, ON] = None):

        if name is None:
            raise ValueError('Node name should not be None.')

        # In a program, each operation node should have a unique name as
        # its identifier.
        self._name = name
        self._depth = depth

        # LHS and RHS values are ordered.
        # input ports are external inputs.
        self._input_ports = OrderedDict(
            input_ports) if input_ports else OrderedDict()

        # a node may produce multiple ordered results.
        # output ports produced values to the external environment.
        self._output_ports = OrderedDict(
            output_ports) if output_ports else OrderedDict()

        self.parents = OrderedDict(parents) if parents else OrderedDict()

    def __str__(self, indent: int = 0):
        indent_str = indent * ' '
        str_to_print = '{}\n'.format(self.__class__.__name__)

        indent_str = indent * ' '
        str_to_print += '{}|- name = {}\n'.format(indent_str, self.name)
        str_to_print += '{}|- opcode = {}\n'.format(indent_str, self.opcode)
        str_to_print += '{}|- depth = {}\n'.format(indent_str, self.depth)
        str_to_print += '{}|- parent = {}\n'.format(
            indent_str, ', '.join([p for p in self.parents]))

        def _print_port(name, storage, indent_str, port_type='in'):
            if storage is not None:
                return '{}|- {} = {} : {}\n'.format(indent_str, port_type,
                                                    name,
                                                    storage.__str__(indent))
            else:
                return '{}|- {} = {} : to-infer\n'.format(
                    indent_str, port_type, name)

        for name, storage in self.input_ports.items():
            str_to_print += _print_port(name, storage, indent_str)

        for name, storage in self.output_ports.items():
            str_to_print += _print_port(name, storage, indent_str, 'out')

        if hasattr(self, '_state_init'):
            for name, storage in self._state_init.items():
                str_to_print += _print_port(name, storage, indent_str,
                                            'state_init')

        if hasattr(self, '_state'):
            for name, storage in self._state.items():
                str_to_print += _print_port(name, storage, indent_str, 'state')

        return str_to_print

    __repr__ = __str__

    @property
    def name(self) -> str:
        return self._name

    @property
    def depth(self) -> int:
        return self._depth

    @depth.setter
    def depth(self, d: int):
        self._depth = d

    @property
    def opcode(self) -> int:
        return NodeBase.opcode

    @property
    def arity(self) -> int:
        return NodeBase.arity

    @property
    def input_ports(self) -> Dict:
        return self._input_ports

    @property
    def output_ports(self) -> Dict:
        return self._output_ports

    def add_input_port(self, name: str = None, storage: Storage = None) -> str:
        """A Port is a pair of <identifier, Storage>. Order is important.

        Returns:
            str, port name.

        If `name` is not given, the default name is <node_name>@in<index>.
        """

        # FIXME(ying): this branch is to be compatible with unittest,
        # delete this branch when parser if completed.
        # name cannot be None
        if name is None:
            name = '{}@in{}'.format(self.name, len(self._input_ports))

        self._input_ports[name] = storage

        return name

    def add_output_port(self, name: str = None,
                        storage: Storage = None) -> str:
        """A Port is a pair of <identifier, Storage>. Order is important.

        Returns:
            str, port name.

        If `name` is not given, the default name is <node_name>@out<index>.
        """

        if name is None:
            name = '{}@out{}'.format(self.name, len(self._output_ports))

        self._output_ports[name] = storage

        return name

    def get_input_port_name(self, i: int) -> str:
        if i >= len(self._input_ports):
            raise IndexError('index is out of boundary.')
        return list(self._input_ports.keys())[i]

    def get_output_port_name(self, i: int) -> str:
        if i >= len(self._output_ports):
            raise IndexError('index is out of boundary.')
        return list(self._output_ports.keys())[i]

    def get_input_storage(self, i: int) -> Storage:
        if i >= len(self._input_ports):
            raise IndexError('index is out of boundary.')
        return self.input_ports[self.get_input_port_name(i)]

    def get_output_storage(self, i: int) -> Storage:
        if i >= len(self._output_ports):
            raise IndexError('index is out of boundary.')
        return self.output_ports[self.get_output_port_name(i)]


class OperationNode(NodeBase, metaclass=ABCMeta):
    """Primitive operations, equivalent to a pure function.

    Further devided into:
    1. compute-intensive arithmetic operations.
        - elementwise
        - reduction
        - broadcast
        - contraction
    2. operations that changes the interpretation of physical memory.
    3. data movement/copy.
    """

    def __init__(self, name: str, input_ports, output_ports, *args, **kwargs):
        super(OperationNode, self).__init__(name, input_ports, output_ports,
                                            *args, **kwargs)

        # ordered keyword argument of a primitive operation.
        # attribute is not differentiable. It is to provide necessary information,
        # like axis, shape, device.
        self.attributes: Dict = {}

    def __str__(self, indent=2):
        str_to_print = super(OperationNode, self).__str__(indent)

        indent_str = indent * ' '
        str_to_print += '{}|- attrs = {}\n'.format(indent_str, self.attributes)

        return str_to_print

    __repr__ = __str__

    # TODO(ying): this method is abstract, uncomment after unify the
    # implementations of tensor primitives.
    @abstractmethod
    def propagate_storage(self):
        assert len(self.input_ports.values()) == self.arity
        assert len(self.output_ports) == 1

        self.check_tensor_storages(*self.input_ports.values())

    def is_inferred(self):
        if all(self.input_ports.values()):
            return True
        else:
            return False

    def check_tensor_storages(self, *storages) -> Storage:
        for s in storages:
            if not isinstance(s, TensorStorage):
                raise TypeError(('Expected TensorStorage, '
                                 f'got {type(s).__name__}.'))

        if self.arity == 1:
            # Unary operator
            return storages[0].element_type
        elif self.arity == 2:
            # Binary operator
            storage1, storage2 = storages
            assert storage1.element_type.is_equal_type(storage2.element_type)
            return storage1.element_type
        elif self.arity == 0:
            return None
        else:
            raise NotImplementedError()


class Elementwise(OperationNode):
    def __init__(self, name, input_ports, output_ports, *args, **kwargs):
        super(Elementwise, self).__init__(name, input_ports, output_ports,
                                          *args, **kwargs)

    def infer_shape(self, shape1: Tuple[int]) -> Tuple[int]:
        return shape1

    def propagate_storage(self):
        super(Elementwise, self).propagate_storage()
        self.check_tensor_storages(*self.input_ports.values())

        s1 = list(self.input_ports.values())[0]

        out_storage = s1.clone()
        out_storage.shape = self.infer_shape(s1.shape)
        self.output_ports[list(self.output_ports.keys())[-1]] = out_storage


class AccessNode(OperationNode):
    """Memory-intensive data movement operations.

    AccessNode is a binary relation over index sets of two ordered collections.
    AccessNodes could be chained.
    """

    def __init__(self, name, input_ports, output_ports, *args, **kwargs):
        super(AccessNode, self).__init__(name, input_ports, output_ports,
                                         *args, **kwargs)

    def check_tensor_storages(self, *storages) -> Storage:
        for s in storages:
            if not isinstance(s, FractalTensorStorage):
                raise TypeError(('Expected FractalTensorStorage, '
                                 f'got {type(s).__name__}.'))

    def propagate_storage(self):
        assert len(self.input_ports.values()) == self.arity
        assert len(self.output_ports) == 1

        self.check_tensor_storages(*self.input_ports.values())


class EdgeEnd(object):
    """An end of a edge is determined by the node name and the port name."""

    def __init__(self, node: str, port: str):
        # NOTE(ying): in the entire graph, each node name and port name
        # MUST be unique. This Ctor cannot gaurantee the uniqueness which
        # SHOULD be gauranteed by the caller.
        self._node = node
        self._port = port

    @property
    def node(self):
        """node is immutable after creation."""
        return self._node

    @property
    def port(self):
        """port is immutable after creation."""
        return self._port

    def update_port(self, n):
        self._port = n

    def __str__(self):
        return '< {} : {} >'.format(self.node, self.port)

    __repr__ = __str__

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.node == other.node and self.port == other.port


class BlockNode(NodeBase):
    """The whole program is block-structured.

    The intermediate representation abstracts away inessential aspects
    of the computational process. Construction of the IR program (parsing)
    performs as many optimizations as possible.

    The IR representation is measured by:
    1. optimizations are inherent
    2. analysis is efficient
    3. transformation is simple

    A BlockNode is a composition of multiple primitive operations. The
    externally-observable behaviors of a BlockNode is like a pure function
    that consumes some immutable values and produces new values.

    1. Operations in a BlockNode share control structure that is the path from
       root node to current block.
    2. A BlockNode may have multiple input ports and multiple output ports.


    Args:
        name, str: a unique identifier of the current node.
        depth, str: depth of the current node.
        input_ports, OrderedDict:
        output_ports, OrderedDict:
    """
    opcode = 'block'
    arity = 0

    def __init__(self, name: str, input_ports: Dict, output_ports: Dict, *args,
                 **kwargs):
        super(BlockNode, self).__init__(
            name,
            input_ports=input_ports,
            output_ports=output_ports,
            *args,
            **kwargs)

        self.nodes: OrderedDict[str, NodeBase] = OrderedDict()

        # TODO(Ying): `_body_in` and `_body_out` have to hold access functions.
        # body_in and body_out are input_ports and output_ports of body
        # of the ParallelNode. They are invisible outside the ParallelNode.

        # BlockNode's body_in port is the same as the input port
        self._body_in = OrderedDict()
        for name, storage in self._input_ports.items():
            self._body_in[name] = storage

        # BlockNode's body_in port is the same as the output port
        self._body_out = OrderedDict()
        for name, storage in self._output_ports.items():
            self._body_out[name] = storage

        # graph stored in the form of adjacency list
        self.edges: Dict[str, List[str]] = {}
        self.in_edges: Dict[str, List[str]] = {}
        self.out_edges: Dict[str, List[str]] = {}

        self._storage_inferred = False

    def __str__(self, indent: int = 0):
        str_to_print = super(BlockNode, self).__str__(indent)

        indent_str = indent * ' '
        nodes_str = ''
        for node in self.nodes:
            nodes_str += '{}({}), '.format(self.nodes[node].name,
                                           self.nodes[node].opcode)
        str_to_print += '{}|- nodes: {}\n'.format(indent_str, nodes_str)

        def _edge_str(edges):
            s = ''
            for tail in edges:
                for head in edges[tail]:
                    s += '{} --> {}, '.format(str(tail), str(head))
            return s

        str_to_print += '{}|- in_edges: {}\n'.format(indent_str,
                                                     _edge_str(self.in_edges))
        str_to_print += '{}|- out_edges: {}\n'.format(
            indent_str, _edge_str(self.out_edges))
        str_to_print += '{}|- edges: {}\n'.format(indent_str,
                                                  _edge_str(self.edges))
        return str_to_print

    __repr__ = __str__

    @property
    def bodyin(self):
        return self._body_in

    @property
    def bodyout(self):
        return self._body_out

    @staticmethod
    def gen_bodyin_port_name(name):
        return name

    @staticmethod
    def gen_bodyout_port_name(name):
        return name

    def dump(self, file_name):
        pass

    def _get_var_info(self, name: str):
        """name is a internally generated name."""

        info = name.split('#')
        if len(info) != 3:
            raise NameError(f'{name} is not a compiler generated name.')
        return info

    def get_bodyin_port_name(self, name) -> str:
        if not (name in self._input_ports):
            raise KeyError((f'Port {name} is not an input port and does not '
                            'have a corresponding bodyin port.'))

        return name

    def get_bodyout_port_name(self, name) -> str:
        if not (name in self._output_ports):
            raise KeyError((f'Port {name} is not an output port and does not '
                            'have a corresponding bodyout port.'))

        return name

    def is_leaf_block(self) -> bool:
        """
        If all the operations in the current BlockNode(self) are primitive
        tensor operations instead of BlockNode, then the current BlockNode
        is a leaf BlockNode.
        """
        for node in self.nodes:
            if isinstance(self.nodes[node], BlockNode):
                return False
        return True

    def get_storage_info_over_edge(self, tail, tip):
        def _tip_port_in_node(target_port: str, node: NodeBase):
            if target_port in node.input_ports:
                return node.input_ports[target_port]
            elif target_port.endswith('@bodyout'):
                return node._body_out[target_port]
            elif hasattr(node, '_state_init'):
                if target_port in node._state_init:
                    return node._state_init[target_port]
                else:
                    raise ValueError()
            if target_port in node.output_ports:
                #FIXME(ying): check why this banch is hit.
                return node.output_ports[target_port]
            else:
                raise ValueError()

        def _str_type_info(storage):
            s = storage.storage if isinstance(storage,
                                              StorageInfoTree) else storage
            if isinstance(s, FractalTensorStorage):
                return 'FT@{}<T<{}, {}>>'.format(
                    s.depth, s.element_shape(),
                    str(s.element_type()._dtype).split()[-1])
            elif isinstance(s, TensorStorage):
                return 'T<{}, {}>'.format(s.shape, str(s._dtype).split()[-1])
            elif isinstance(s, Number):
                return str(s)
            else:
                return 'Unknown'
                #raise TypeError()

        if tip.node == self.name:
            return _str_type_info(_tip_port_in_node(tip.port, self))

        if tip.node in self.nodes:
            return _str_type_info(
                _tip_port_in_node(tip.port, self.nodes[tip.node]))
        else:
            # FIXME(ying): unify the logic. If tip node is not in the current
            # block, this branch is hit.
            raise ValueError()

    def add_input_node(self, node: NodeBase,
                       *in_edges: Tuple[Tuple[str, str]]):
        """
        If a node is specified as the input node of a ParallelNode, special
        edges (flow of data) shown below are automatically added: from one of
        the body_in ports of the ParallelNode to one of the input port of
        `node`:

                      body_in_port(self) --> input_port(node)
        Args:
            node, NodeBase: node to be added into the current ParallelNode.
            in_edges, List[Tuple[str, str]]: List of port pairs. Each pair
                      indicates a flow of data from one of the input
                      ports of the current BlockNode to one of the input port
                      of the `node` which resides in the current BlockNode.
                      Port are specified by name.
        Example:


        """
        self.add_node(node)

        for edge in in_edges:
            tail_port, head_port = edge
            tail_port = self.get_bodyin_port_name(tail_port)

            if not head_port in node.input_ports:
                if hasattr(node, '_state_init'):
                    # FIXME(ying): hard-coded way to test the current object
                    # is an AggregateNode.
                    if not head_port in node._state_init:
                        raise ValueError(
                            f'{head_port} is not a port of Node:{node}.')
                else:
                    raise ValueError((f'{head_port} is not an '
                                      f'input port of {node.name}'))

            tail = EdgeEnd(node=self.name, port=tail_port)
            head = EdgeEnd(node=node.name, port=head_port)

            if not (tail in self.in_edges):
                self.in_edges[tail] = [head]
            else:
                self.in_edges[tail].append(head)

    def add_output_node(self, node: NodeBase,
                        *out_edges: Tuple[Tuple[str, str]]):
        """
        If a node is specified as the output node of a ParallelNode, special
        edges (flow of data) shown below are automatically added: from one of
        the output port of `node` to body_out ports of the ParallelNode:

                      output_port(node) --> body_out_port(self)
        Args:
            node, NodeBase: node to be added into the current ParallelNode.
            out_edges, Tuple[Tuple[str, str]]: List of port pairs. Each pair
                      indicates a flow of data from one of the output ports
                      of `node` which reside in the current ParallelNode
                      to one of the body_out port of the current ParallelNode.
                      Port are specified by name.
        """

        self.add_node(node)

        for edge in out_edges:
            tail_port, head_port = edge
            head_port = self.get_bodyout_port_name(head_port)

            head = EdgeEnd(node=self.name, port=head_port)

            if not tail_port in node.output_ports:
                raise ValueError((f'{tail_port} is not an '
                                  f'output port of {node.name}.'))
            tail = EdgeEnd(node=node.name, port=tail_port)

            if not (tail in self.out_edges):
                self.out_edges[tail] = [head]
            else:
                self.out_edges[tail].append(head)

    def add_node(self, node: NodeBase):
        if not (isinstance(node, NodeBase) or isinstance(node, ValueNode)):
            raise TypeError('Expected NodeBase or ValueNode.')

        node.parents[self.name] = self
        self.nodes[node.name] = node

        depth = self.depth + 1
        node.depth = max(depth, node.depth)

    def _check_tail(self, node: NodeBase, port: str):
        def _check_port(node: NodeBase, target_port: str):
            if not target_port in node.output_ports:
                if isinstance(node, BlockNode):
                    if target_port in node._body_in:
                        pass
                    elif target_port in node.output_ports:
                        pass
                    elif hasattr(node, '_state'):
                        # an Aggregate node
                        if not (target_port in node._state):
                            raise ValueError(
                                (f'Tail port: {port} should be an '
                                 'state_in port.'))
                    else:
                        raise ValueError(
                            f'Tail port: {port} should be an body_in port.')
                else:
                    raise ValueError(
                        f'Tail port: {port} should be an output port.')

        if node.name == self.name:
            _check_port(self, port)
        else:
            if node.name in self.nodes:
                _check_port(self.nodes[node.name], port)
            else:
                # FIXME(ying): check `node` is a parent of `self`.
                _check_port(node, port)

    def _check_tip(self, node: NodeBase, port: str):
        """
        tip.node MUST be inside the current BlockNode.
        tip.port MUST either be an:
            (1) input port,
         or (2) the _state_init port of an AggregateNode.
        """

        if not node.name in self.nodes:
            raise KeyError(
                f'Tip: {node.name} is not inside BlockNode: {self.name}.')

        target_node = self.nodes[node.name]
        if not (port in target_node.input_ports):
            if isinstance(target_node, ParallelNode) and hasattr(
                    node, '_state_init'):
                if not (port in node._state_init):
                    raise ValueError(
                        (f'Tip: {node.name} can only flow '
                         'into _state_init port of an AggregateNode.'))
            else:
                raise ValueError(f'Tip port {port} is not an input port.')

    def add_edge(self,
                 tail: Tuple[NodeBase, str],
                 tip: Tuple[NodeBase, str],
                 edge_type=None,
                 check_edge=True):
        """
        Args:
            edge_type: in, out, or internal.
        """
        if check_edge:
            # FIXME(ying): this branch is a hotfix.
            # In the condition that, `tail` is not in the current block, but
            # parent block, `tail` is checked by the caller.
            self._check_tail(*tail)
            self._check_tip(*tip)

        node, port = tail
        e1 = EdgeEnd(node.name, port)

        node, port = tip
        e2 = EdgeEnd(node.name, port)

        if edge_type == 'in':
            if e1 not in self.in_edges:
                self.in_edges[e1] = []
            self.in_edges[e1].append(e2)
        elif edge_type == 'out':
            if e1 not in self.out_edges:
                self.out_edges[e1] = []
            self.out_edges[e1].append(e2)
        else:
            if e1 not in self.edges:
                self.edges[e1] = []
            self.edges[e1].append(e2)

    def update_edge_tip(self, edge_type: str, old_tip: EdgeEnd,
                        new_tip: EdgeEnd) -> bool:
        edges = None
        if edge_type == 'in':
            edges = self.in_edges
        elif edge_type == 'out':
            edges = self.out_edges
        elif edge_type == 'internal':
            edges = self.edges
        else:
            raise ValueError(("Error edge types, should be one of "
                              "'in', 'out', or 'internal', got f{edge_type}"))

        for tail in edges:
            for tip in edges[tail]:
                if tip.node == old_tip.node and tip.port == old_tip.port:
                    items = edges[tail]
                    idx = items.index(old_tip)
                    items.remove(old_tip)
                    del edges[tail]
                    items.insert(idx, new_tip)
                    edges[tail] = items
                    return True
        return False

    def add_input_port(self, name: str, storage: Storage = None):
        """Add an input port to the BlockNode.

        Args:
            name, str: name of the input port.
            storage, Storage: the Storage information of the port.

        NOTE:
            a corresponding bodyin port is automatically added also.
        """
        if name in self._input_ports:
            raise KeyError(f'Duplicated port name {name}.')
        self._input_ports[name] = storage

        n = self.gen_bodyin_port_name(name)
        if n in self._body_in:
            raise KeyError(f'Duplicated name for the bodyin port: {name}.')

        self._body_in[n] = None

    def add_output_port(self, name: str, storage: Storage = None):
        """Add an output port to the BlockNode.

        Args:
            name, str: name of the input port.
            storage, Storage: the Storage information of the port.

        NOTE:
            a corresponding bodyout port is automatically added also.
        """
        if name in self._output_ports:
            raise KeyError(f'Duplicated port name {name}.')
        self._output_ports[name] = storage

        n = self.gen_bodyout_port_name(name)
        if n in self._body_out:
            raise KeyError(f'Duplicated name for the bodyout port: {name}.')

        self._body_out[n] = None

    def search_output_node(self):
        """
        A primitive tensor operation is inside the parallel pattern.
        No return statement is explictly used to denote the output
        node of a parallel pattern.

        In Python, a lambda function is a single-line function which
        contains only a single expression.
        """
        # FIXME(ying): use BFS to determine the output ports.
        return list(self.nodes.values())[0]

    def _to_storage(self, s):
        if isinstance(s, Storage):
            return s
        elif isinstance(s, StorageInfoTree):
            return s.storage
        else:
            raise TypeError()

    def propagate_over_in_edges(self):
        # IR nodes whose type information is not fully inferred.
        partial_nodes = set()

        # an `in_edge` flows from a `input port`(tail) of the BlockNode
        # to a `input_port`(tip) of an OperationNode inside the BlockNode.
        for tail in self.in_edges:
            for tip in self.in_edges[tail]:
                tip_node = self.nodes[tip.node]
                if tail.port in self.input_ports:
                    tip_node.input_ports[tip.port] = self._to_storage(
                        self.input_ports[tail.port])
                else:
                    raise KeyError(
                        (f'{tail.port} is not an input '
                         f'port of the current BlockNode {self.name}.'))

                if tip_node.is_inferred():
                    tip_node.propagate_storage()
                    partial_nodes.discard(tip.node)
                else:
                    partial_nodes.add(tip.node)
        return partial_nodes

    def propagate_over_internal_edges(self, partial_nodes):
        # Storage information flows from the output_port of an `input_nodes`
        # to an `input_port` of an internal node.
        for tail in self.edges:
            for tip in self.edges[tail]:
                in_storage = None
                if tail.node in self.nodes:
                    tail_node = self.nodes[tail.node]
                    assert tail_node.is_inferred()
                    if tail_node.output_ports[tail.port] is None:
                        # constant operations, like zeros, ones hit this branch.
                        tail_node.propagate_storage()
                    in_storage = tail_node.output_ports[tail.port]
                elif tail.node == self.name:
                    if tail.port in self.input_ports:
                        in_storage = self.input_ports[tail.port]
                    elif hasattr(self, '_state') and tail.port in self._state:
                        in_storage = self._state[tail.port]
                    else:
                        raise ParseError('Error tail.')
                else:
                    raise NotImplementedError()
                assert in_storage

                tip_node = self.nodes[tip.node]
                if tip.port in tip_node.input_ports:
                    tip_node.input_ports[tip.port] = in_storage
                elif hasattr(
                        tip_node,
                        '_state_init') and tip.port in tip_node._state_init:
                    tip_node._state_init[tip.port] = in_storage
                    idx = list(tip_node._state_init.keys()).index(tip.port)
                    tip_node._state[list(
                        tip_node._state.keys())[idx]] = in_storage
                elif hasattr(tip_node,
                             '_state') and tip.port in tip_node._state:
                    tip_node._state[tip.port] = in_storage
                else:
                    raise ParseError('Error tip port.')

                if tip_node.is_inferred():
                    tip_node.propagate_storage()
                    partial_nodes.discard(tip.node)
                else:
                    partial_nodes.add(tip.node)

        return partial_nodes

    def propagate_over_out_edges(self):
        output_nodes = {}
        # an `out_edge` flows from the `output port`(tail) of an internal_node
        # inside the BlockNode to `body_out`(tip) of the BlockNode.
        for tail in self.out_edges:
            for tip in self.out_edges[tail]:
                assert tip.node == self.name

                storage = self.nodes[tail.node].output_ports[tail.port]

                if self.opcode == 'block':
                    self.output_ports[tip.port] = storage
                else:
                    assert tip.port.endswith('@bodyout')
                    self.bodyout[tip.port] = storage
                    outport_name = tip.port.split('@bodyout')[0]

                    self.output_ports[outport_name] = FractalTensorStorage(
                        storage)

    def propagate_storage(self):
        """propagate storage information throughout the BlockNode."""
        assert all(self.input_ports.values())

        partial_nodes = self.propagate_over_in_edges()
        partial_nodes = self.propagate_over_internal_edges(partial_nodes)

        if len(partial_nodes):
            raise ParseError(('All nodes should be infered '
                              'befor propagating attributes to out edges.'))
        self.propagate_over_out_edges()

    def increase_depth(self):
        """Increase the nesting depth of the block."""

        for _, node in self.nodes.items():
            if isinstance(node, BlockNode):
                node.increase_depth()
            node.depth += 1


class ParallelNode(BlockNode):
    """
    Data parallel patterns are operations to create new FractalTensor values.
    ParallelNodes dictate the strict orders of read/write accessing to a
    large contiguous memory (in form of FractalTensor values). The order is
    important.
    """

    opcode = ''
    arity = 0

    def __init__(self, name: str, input_ports: Dict, output_ports: Dict, *args,
                 **kwargs):
        super(ParallelNode, self).__init__(name, input_ports, output_ports,
                                           *args, **kwargs)

        self._body_in.clear()
        for name, storage in self._input_ports.items():
            # each input port has a corresponding port called `body_in` whose
            # storage is the elementary type of the input port.
            bodyin_name = self.get_bodyin_port_name(name)
            self._body_in[bodyin_name] = storage._dtype if storage else storage

        self._body_out.clear()
        for name, storage in self._output_ports.items():
            # each output port has a corresponding port called `body_out` whose
            # storage is the elementary type of the output port.
            bodyout_name = self.get_bodyout_port_name(name)
            self._body_out[
                bodyout_name] = storage._dtype if storage else storage

    @staticmethod
    def gen_bodyin_port_name(name) -> str:
        return '{}@bodyin'.format(name)

    @staticmethod
    def gen_bodyout_port_name(name) -> str:
        return '{}@bodyout'.format(name)

    def get_bodyin_port_name(self, name) -> str:
        """
        Given the name of an input port, returns the name of the
        corresponding body_in port.

        Return None if `name` is not an input port of the current node.
        """

        port_name = self.gen_bodyin_port_name(name)
        if not (port_name in self._body_in):
            logging.warning(f'The bodyin port {name} is not created yet.')
        return port_name

    def get_bodyout_port_name(self, name) -> str:
        """
        Given the name of an output port, returns the name of the
        corresponding body_out port.

        Return None if `name` is not an output port of the current node.
        """

        port_name = self.gen_bodyout_port_name(name)
        if not (port_name in self._body_out):
            logging.warning(f'The bodyout port "{name}" is not created yet.')
        return port_name

    def retrieve_tail_port_storage(self, parents: OrderedDict, tail_node: str,
                                   tail_port: str) -> Storage:
        assert len(parents)
        for p in parents:
            if p != tail_node:
                continue

            node = parents[p]

            if node.opcode == 'block':
                if tail_port in node.input_ports:
                    return node.input_ports[tail_port]
                else:
                    raise NotImplementedError()
            elif isinstance(node, Aggregate):
                if tail_port.endswith('@bodyin'):
                    assert tail_port in node.bodyin

                    inport_name = tail_port.split('@bodyin')[0]

                    idx = 0
                    for i, k in enumerate(node.input_ports.keys()):
                        if k == tail_port:
                            idx = i
                            break
                    storage = node.input_ports[list(
                        node.input_ports.keys())[idx]]

                    node.bodyin[tail_port] = storage._dtype
                    return storage._dtype
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

        raise ValueError(("Error tail node and tail port, "
                          "which should be defined in parents' scope."))

    @abstractmethod
    def is_inferred(self):
        pass

    def propagate_over_in_edges(self):
        # IR nodes whose type information is not fully inferred.
        partial_nodes = set()

        # an `in_edge` flows from a `input port`(tail) of the BlockNode
        # to a `input_port`(tip) of an OperationNode inside the BlockNode.
        in_storage = None
        for tail in self.in_edges:
            for tip in self.in_edges[tail]:
                tip_node = self.nodes[tip.node]

                if tail.port in self.input_ports:
                    in_storage = self._to_storage(self.input_ports[tail.port])
                elif tail.port in self.bodyin:
                    inport_name = tail.port.split('@bodyin')[0]
                    in_storage = self.input_ports[inport_name]._dtype
                    self.bodyin[tail.port] = in_storage
                elif hasattr(self, '_state') and tail.port in self._state:
                    in_storage = self._state[tail.port]
                else:
                    in_storage = self.retrieve_tail_port_storage(
                        self.parents, tail.node, tail.port)

                if in_storage is None:
                    raise ParseError(f'{tail.port} is not inferred yet.')

                if tip.port in tip_node.input_ports:
                    tip_node.input_ports[tip.port] = in_storage
                elif hasattr(
                        tip_node,
                        '_state_init') and tip.port in tip_node._state_init:
                    tip_node._state_init[tip.port] = in_storage

                    idx = list(tip_node._state_init.keys()).index(tip.port)
                    tip_node._state[list(
                        tip_node._state.keys())[idx]] = in_storage
                else:
                    raise ParseError('Error tip port.')

                if tip_node.is_inferred():
                    tip_node.propagate_storage()
                    partial_nodes.discard(tip.node)
                else:
                    partial_nodes.add(tip.node)
        return partial_nodes


class BranchNode(NodeBase):
    """Conditional branching.

    Conditional branching appears only inside a ParallelNode (inside a loop).
    It cannot appear at the outermost control level.
    """

    def __init__(self, name: str, depth: int = -1, *args, **kwargs):
        super(BranchNode, self).__init__(name, depth, args, kwargs)

        self.predicate: List[OperationNode] = []

        self.true_branch: Union[BlockNode, ParallelNode] = None
        self.false_branch: Union[BlockNode, ParallelNode] = None


class ApplyToEach(ParallelNode):
    opcode = None
    arity = 0

    def __init__(self, name: str, opcode: str, input_ports: Dict,
                 output_ports: Dict, *args, **kwargs):
        if opcode not in kaleido.parser.context._APPLY_TO_EACH:
            raise ValueError(f'Unsupported pattern: {opcode}.')

        self.opcode = opcode
        super(ApplyToEach, self).__init__(name, input_ports, output_ports,
                                          *args, **kwargs)

    def is_inferred(self):
        if all(self.input_ports.values()):
            return True
        else:
            return False


class Aggregate(ParallelNode):
    opcode = None
    arity = 0

    def __init__(self,
                 name,
                 opcode: str,
                 input_ports: Dict,
                 output_ports: Dict,
                 state_init: Dict = None,
                 *args,
                 **kwargs):
        if opcode not in kaleido.parser.context._AGGREGATE:
            raise ValueError(f'Unsupported pattern: {opcode}.')
        self.opcode = opcode
        super(Aggregate, self).__init__(name, input_ports, output_ports, *args,
                                        **kwargs)

        # stores state initializer.
        self._state_init = OrderedDict() if state_init is None else state_init

        self._state = OrderedDict()
        for port in self.output_ports:
            self.add_state_port(self.gen_state_port_name(port), storage=None)

        # in parsing, rhs is parsed first, at that time, output port (lhs) is
        # not created yet. store the output nodes.
        self._tmp_out_nodes = []

    @staticmethod
    def gen_state_port_name(name: str) -> str:
        return '{}@state'.format(name)

    def get_state_port_name(self, i: int):
        """Get the internal state_in port's name."""

        if i >= len(self._state):
            raise IndexError('index out of boundary.')

        return list(self._state.keys())[i]

    def get_state_init_port_name(self, i: int):
        """Get the internal state_in port's name."""

        if i >= len(self._state_init):
            raise IndexError('index out of boundary.')

        return list(self._state_init.keys())[i]

    def add_state_port(self, name: str, storage: Storage = None):
        if name in self._state:
            raise KeyError(f'Duplicated name: {name}.')

        self._state[name] = storage

    def add_state_init_port(self, name: str, storage: Storage = None):
        if name in self._state_init:
            raise KeyError(f'Duplicated name: {name}.')

        self._state_init[name] = storage

    def is_inferred(self):
        if (all(self.input_ports.values()) and all(self._state_init.values())):
            return True
        else:
            return False
