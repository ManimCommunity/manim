from .types.vectorized_mobject import VMobject
from .geometry import Dot, Line

from manim import UP

import networkx as nx
import numpy as np


class Graph(VMobject):
    def __init__(self, vertices, edges, scale=1):
        VMobject.__init__(self)

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(vertices)
        nx_graph.add_edges_from(edges)
        self._graph = nx_graph
        self._layout = nx.layout.spring_layout(nx_graph)

        self.vertices = dict([(v, Dot()) for v in vertices])
        for v in self.vertices:
            self[v].move_to(np.append(self._layout[v], [0]))

        self.edges = dict(
            [
                ((u, v), Line(self[u].get_center(), self[v].get_center()))
                for (u, v) in edges
            ]
        )

        self.add(*self.vertices.values())
        self.add(*self.edges.values())

        for (u, v), edge in self.edges.items():

            def update_edge(e, u=u, v=v):
                e.set_start_and_end_attrs(self[u].get_center(), self[v].get_center())
                e.generate_points()

            update_edge(edge)
            edge.add_updater(update_edge)

        print(self.submobjects)

    def __getitem__(self, v):
        return self.vertices[v]
