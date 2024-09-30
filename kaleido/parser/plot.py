__all__ = [
    'PlotProgram',
]
from graphviz import Digraph

import kaleido

from kaleido.frontend.types import StorageInfoTree
from kaleido.frontend.types import TensorStorage
from kaleido.frontend.types import FractalTensorStorage
from kaleido.parser.ir_nodes import BlockNode
from kaleido.parser.ir_nodes import ParallelNode


class PlotProgram(object):
    def __init__(self, graph_name='example'):
        self.g = Digraph(graph_name)
        self.count = 0  # generate internal name

    def _gen_name(self):
        n = '_tmp{}'.format(self.count)
        self.count += 1
        return n

    def str_type_info(self, storage):
        s = storage
        if isinstance(s, FractalTensorStorage):
            return 'FT@{}<T<{}, {}>>'.format(
                s.depth, s.element_shape(),
                str(s.element_type()._dtype).split()[-1])
        elif isinstance(s, TensorStorage):
            return 'T<{}, {}>'.format(s.shape, str(s._dtype).split()[-1])
        else:
            raise TypeError()

    def plot_program_inputs(self, root, graph):
        for x in root.input_ports:
            type_label = self.str_type_info(root.input_ports[x])
            type_name = self._gen_name()
            graph.node(
                type_name,
                type_label,
                shape='rect',
                style='filled',
                fontsize='8')
            graph.edge(type_name, root.name, label=x, fontsize='9')

    def plot_block_in_edges(self, root, graph):
        """
        The special edge: from one of the block's input ports to an
        input port of one of the block's internal node.
        """

        style = 'solid'
        for tail in root.in_edges:
            for tip in root.in_edges[tail]:
                type_label = root.get_storage_info_over_edge(tail, tip)

                if tail.port.endswith('@bodyin'):
                    # The BlockNode is a ParallelNode, access function is
                    # attached to the in_edge.
                    graph.edge(
                        tail.node,
                        tip.node,
                        fontsize='8',
                        label='{}\n{}'.format(tail.port, type_label),
                        color='firebrick',
                        style='dotted')
                else:
                    graph.edge(
                        tail.node,
                        tip.node,
                        fontsize='8',
                        label='{}\n{}'.format(tail.port, type_label))
        if root.depth == 0:
            # the entire program is a function, plots the inputs to
            # of the program.
            self.plot_program_inputs(root, graph)

    def plot_block_out_edges(self, root, graph):
        """
        The special edge: from an output port of one of the block's internal
        node to one of the block's output ports.
        """
        for tail in root.out_edges:
            if isinstance(root.nodes[tail.node], BlockNode):
                return

        for tail in root.out_edges:
            for tip in root.out_edges[tail]:
                type_name = self._gen_name()
                type_label = root.get_storage_info_over_edge(tail, tip)

                label = tail.port
                color = 'black'
                style = 'solid'

                if tip.port.endswith('@bodyout'):
                    # The BlockNode is a ParallelNode
                    label = '{}\nbody out'.format(label)
                    color = 'deepskyblue'
                    style = 'dotted'

                graph.node(
                    type_name,
                    type_label,
                    shape='rect',
                    style='filled',
                    fontsize='8',
                    fillcolor='lightskyblue2')
                graph.edge(
                    tail.node,
                    type_name,
                    fontsize='8',
                    label=label,
                    style=style,
                    color=color)

    def plot_block_edges(self, root, graph):
        for tail in root.edges:
            for tip in root.edges[tail]:
                type_label = root.get_storage_info_over_edge(tail, tip)
                if tail.port.endswith('@state'):
                    graph.edge(
                        tail.node,
                        tip.node,
                        fontsize='8',
                        color='darkorange',
                        label='{}\n{}'.format(tail.port, type_label))
                else:
                    graph.edge(
                        tail.node,
                        tip.node,
                        fontsize='8',
                        label='{}\n{}'.format(tail.port, type_label))

    def plot_a_level(self, root, graph):
        # 1. plot nodes in the BlockNode.
        style = None
        for name, node in root.nodes.items():
            label = '{}\n{}, depth={}'.format(node.name, node.opcode,
                                              node.depth)
            if isinstance(node, BlockNode):
                graph.node(
                    node.name,
                    label,
                    style='filled',
                    color='darkseagreen',
                    fontsize='10')
            else:
                graph.node(node.name, label, fontsize='10')

        # 2. plot edges in the BlockNode.
        if isinstance(root, BlockNode):
            self.plot_block_in_edges(root, graph)
            self.plot_block_edges(root, graph)

        # Recursively plot BlockNodes in current BlockNode
        for name, node in root.nodes.items():
            if isinstance(node, BlockNode):
                self.plot_a_level(node, graph)

        self.plot_block_out_edges(root, graph)

    def plot(self, root: BlockNode):
        label = '{}\n{}, depth={}'.format(root.name, root.opcode, root.depth)
        self.g.node(
            root.name,
            label,
            shape='doubleoctagon',
            style='filled',
            fillcolor='lightsteelblue',
            fontsize='12')

        self.plot_a_level(root, self.g)
        self.g.render()
