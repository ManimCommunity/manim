"""Mobjects used to represent mathematical graphs (think graph theory, not plotting)."""

__all__ = [
    "Graph",
]

from copy import copy
from typing import Hashable, List, Optional, Tuple, Type, Union

import networkx as nx
import numpy as np

from manim.mobject.opengl_mobject import OpenGLMobject

from ..animation.composition import AnimationGroup
from ..animation.creation import Create, Uncreate
from ..utils.color import BLACK
from .geometry import Dot, LabeledDot, Line
from .mobject import Mobject, override_animate
from .opengl_compatibility import ConvertToOpenGL
from .svg.tex_mobject import MathTex
from .types.vectorized_mobject import VMobject


def _determine_graph_layout(
    nx_graph: nx.classes.graph.Graph,
    layout: Union[str, dict] = "spring",
    layout_scale: float = 2,
    layout_config: Union[dict, None] = None,
    partitions: Union[List[List[Hashable]], None] = None,
    root_vertex: Union[Hashable, None] = None,
) -> dict:
    automatic_layouts = {
        "circular": nx.layout.circular_layout,
        "kamada_kawai": nx.layout.kamada_kawai_layout,
        "planar": nx.layout.planar_layout,
        "random": nx.layout.random_layout,
        "shell": nx.layout.shell_layout,
        "spectral": nx.layout.spectral_layout,
        "partite": nx.layout.multipartite_layout,
        "tree": _tree_layout,
        "spiral": nx.layout.spiral_layout,
        "spring": nx.layout.spring_layout,
    }

    custom_layouts = ["random", "partite", "tree"]

    if layout_config is None:
        layout_config = {}

    if isinstance(layout, dict):
        return layout
    elif layout in automatic_layouts and layout not in custom_layouts:
        auto_layout = automatic_layouts[layout](
            nx_graph, scale=layout_scale, **layout_config
        )
        return {k: np.append(v, [0]) for k, v in auto_layout.items()}
    elif layout == "tree":
        return _tree_layout(
            nx_graph,
            root_vertex=root_vertex,
            scale=layout_scale,
        )
    elif layout == "partite":
        if partitions is None or len(partitions) == 0:
            raise ValueError(
                "The partite layout requires the 'partitions' parameter to contain the partition of the vertices",
            )
        partition_count = len(partitions)
        for i in range(partition_count):
            for v in partitions[i]:
                if nx_graph.nodes[v] is None:
                    raise ValueError(
                        "The partition must contain arrays of vertices in the graph",
                    )
                nx_graph.nodes[v]["subset"] = i
        # Add missing vertices to their own side
        for v in nx_graph.nodes:
            if "subset" not in nx_graph.nodes[v]:
                nx_graph.nodes[v]["subset"] = partition_count

        auto_layout = automatic_layouts["partite"](
            nx_graph, scale=layout_scale, **layout_config
        )
        return {k: np.append(v, [0]) for k, v in auto_layout.items()}
    elif layout == "random":
        # the random layout places coordinates in [0, 1)
        # we need to rescale manually afterwards...
        auto_layout = automatic_layouts["random"](nx_graph, **layout_config)
        for k, v in auto_layout.items():
            auto_layout[k] = 2 * layout_scale * (v - np.array([0.5, 0.5]))
        return {k: np.append(v, [0]) for k, v in auto_layout.items()}
    else:
        raise ValueError(
            f"The layout '{layout}' is neither a recognized automatic layout, "
            "nor a vertex placement dictionary.",
        )


def _tree_layout(
    G: nx.classes.graph.Graph,
    root_vertex: Union[Hashable, None],
    scale: float,
) -> dict:
    result = {root_vertex: np.array([0, 0, 0])}

    if not nx.is_tree(G):
        raise ValueError("The tree layout must be used with trees")
    if root_vertex is None:
        raise ValueError("The tree layout requires the root_vertex parameter")

    def _recursive_position_for_row(
        G: nx.classes.graph.Graph,
        result: dict,
        two_rows_before: List[Hashable],
        last_row: List[Hashable],
        current_height: float,
    ):
        new_row = []
        for v in last_row:
            for x in G.neighbors(v):
                if x not in two_rows_before:
                    new_row.append(x)

        new_row_length = len(new_row)

        if new_row_length == 0:
            return

        if new_row_length == 1:
            result[new_row[0]] = np.array([0, current_height, 0])
        else:
            for i in range(new_row_length):
                result[new_row[i]] = np.array(
                    [-1 + 2 * i / (new_row_length - 1), current_height, 0],
                )

        _recursive_position_for_row(
            G,
            result,
            two_rows_before=last_row,
            last_row=new_row,
            current_height=current_height + 1,
        )

    _recursive_position_for_row(
        G,
        result,
        two_rows_before=[],
        last_row=[root_vertex],
        current_height=1,
    )

    height = max(map(lambda v: result[v][1], result))

    return {
        v: np.array([pos[0], 1 - 2 * pos[1] / height, pos[2]]) * scale / 2
        for v, pos in result.items()
    }


class Graph(VMobject, metaclass=ConvertToOpenGL):
    """An undirected graph (that is, a collection of vertices connected with edges).

    Graphs can be instantiated by passing both a list of (distinct, hashable)
    vertex names, together with list of edges (as tuples of vertex names). See
    the examples below for details.

    .. note::

        This implementation uses updaters to make the edges move with
        the vertices.

    Parameters
    ----------

    vertices
        A list of vertices. Must be hashable elements.
    edges
        A list of edges, specified as tuples ``(u, v)`` where both ``u``
        and ``v`` are vertices.
    labels
        Controls whether or not vertices are labeled. If ``False`` (the default),
        the vertices are not labeled; if ``True`` they are labeled using their
        names (as specified in ``vertices``) via :class:`~.MathTex`. Alternatively,
        custom labels can be specified by passing a dictionary whose keys are
        the vertices, and whose values are the corresponding vertex labels
        (rendered via, e.g., :class:`~.Text` or :class:`~.Tex`).
    label_fill_color
        Sets the fill color of the default labels generated when ``labels``
        is set to ``True``. Has no effect for other values of ``labels``.
    layout
        Either one of ``"spring"`` (the default), ``"circular"``, ``"kamada_kawai"``,
        ``"planar"``, ``"random"``, ``"shell"``, ``"spectral"``, ``"spiral"``, ``"tree"``, and ``"partite"``
        for automatic vertex positioning using ``networkx``
        (see `their documentation <https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout>`_
        for more details), or a dictionary specifying a coordinate (value)
        for each vertex (key) for manual positioning.
    layout_scale
        The scale of automatically generated layouts: the vertices will
        be arranged such that the coordinates are located within the
        interval ``[-scale, scale]``. Default: 2.
    layout_config
        Only for automatically generated layouts. A dictionary whose entries
        are passed as keyword arguments to the automatic layout algorithm
        specified via ``layout`` of``networkx``.
    vertex_type
        The mobject class used for displaying vertices in the scene.
    vertex_config
        Either a dictionary containing keyword arguments to be passed to
        the class specified via ``vertex_type``, or a dictionary whose keys
        are the vertices, and whose values are dictionaries containing keyword
        arguments for the mobject related to the corresponding vertex.
    vertex_mobjects
        A dictionary whose keys are the vertices, and whose values are
        mobjects to be used as vertices. Passing vertices here overrides
        all other configuration options for a vertex.
    edge_type
        The mobject class used for displaying edges in the scene.
    edge_config
        Either a dictionary containing keyword arguments to be passed
        to the class specified via ``edge_type``, or a dictionary whose
        keys are the edges, and whose values are dictionaries containing
        keyword arguments for the mobject related to the corresponding edge.

    Examples
    --------

    First, we create a small graph and demonstrate that the edges move
    together with the vertices.

    .. manim:: MovingVertices

        class MovingVertices(Scene):
            def construct(self):
                vertices = [1, 2, 3, 4]
                edges = [(1, 2), (2, 3), (3, 4), (1, 3), (1, 4)]
                g = Graph(vertices, edges)
                self.play(Create(g))
                self.wait()
                self.play(g[1].animate.move_to([1, 1, 0]),
                          g[2].animate.move_to([-1, 1, 0]),
                          g[3].animate.move_to([1, -1, 0]),
                          g[4].animate.move_to([-1, -1, 0]))
                self.wait()

    There are several automatic positioning algorithms to choose from:

    .. manim:: GraphAutoPosition
        :save_last_frame:

        class GraphAutoPosition(Scene):
            def construct(self):
                vertices = [1, 2, 3, 4, 5, 6, 7, 8]
                edges = [(1, 7), (1, 8), (2, 3), (2, 4), (2, 5),
                         (2, 8), (3, 4), (6, 1), (6, 2),
                         (6, 3), (7, 2), (7, 4)]
                autolayouts = ["spring", "circular", "kamada_kawai",
                               "planar", "random", "shell",
                               "spectral", "spiral"]
                graphs = [Graph(vertices, edges, layout=lt).scale(0.5)
                          for lt in autolayouts]
                r1 = VGroup(*graphs[:3]).arrange()
                r2 = VGroup(*graphs[3:6]).arrange()
                r3 = VGroup(*graphs[6:]).arrange()
                self.add(VGroup(r1, r2, r3).arrange(direction=DOWN))

    Vertices can also be positioned manually:

    .. manim:: GraphManualPosition
        :save_last_frame:

        class GraphManualPosition(Scene):
            def construct(self):
                vertices = [1, 2, 3, 4]
                edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
                lt = {1: [0, 0, 0], 2: [1, 1, 0], 3: [1, -1, 0], 4: [-1, 0, 0]}
                G = Graph(vertices, edges, layout=lt)
                self.add(G)

    The vertices in graphs can be labeled, and configurations for vertices
    and edges can be modified both by default and for specific vertices and
    edges.

    .. note::

        In ``edge_config``, edges can be passed in both directions: if
        ``(u, v)`` is an edge in the graph, both ``(u, v)`` as well
        as ``(v, u)`` can be used as keys in the dictionary.

    .. manim:: LabeledModifiedGraph
        :save_last_frame:

        class LabeledModifiedGraph(Scene):
            def construct(self):
                vertices = [1, 2, 3, 4, 5, 6, 7, 8]
                edges = [(1, 7), (1, 8), (2, 3), (2, 4), (2, 5),
                         (2, 8), (3, 4), (6, 1), (6, 2),
                         (6, 3), (7, 2), (7, 4)]
                g = Graph(vertices, edges, layout="circular", layout_scale=3,
                          labels=True, vertex_config={7: {"fill_color": RED}},
                          edge_config={(1, 7): {"stroke_color": RED},
                                       (2, 7): {"stroke_color": RED},
                                       (4, 7): {"stroke_color": RED}})
                self.add(g)

    You can also lay out a partite graph on columns by specifying
    a list of the vertices on each side and choosing the partite layout.

    .. note::

        All vertices in your graph which are not listed in any of the partitions
        are collected in their own partition and rendered in the rightmost column.

    .. manim:: PartiteGraph
        :save_last_frame:

        import networkx as nx

        class PartiteGraph(Scene):
            def construct(self):
                G = nx.Graph()
                G.add_nodes_from([0, 1, 2, 3])
                G.add_edges_from([(0, 2), (0,3), (1, 2)])
                graph = Graph(list(G.nodes), list(G.edges), layout="partite", partitions=[[0, 1]])
                self.play(Create(graph))

    The custom tree layout can be used to show the graph
    by distance from the root vertex. You must pass the root vertex
    of the tree.

    .. manim:: Tree

        import networkx as nx

        class Tree(Scene):
            def construct(self):
                G = nx.Graph()

                G.add_node("ROOT")

                for i in range(5):
                    G.add_node("Child_%i" % i)
                    G.add_node("Grandchild_%i" % i)
                    G.add_node("Greatgrandchild_%i" % i)

                    G.add_edge("ROOT", "Child_%i" % i)
                    G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
                    G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)

                self.play(Create(
                    Graph(list(G.nodes), list(G.edges), layout="tree", root_vertex="ROOT")))
    """

    def __init__(
        self,
        vertices: List[Hashable],
        edges: List[Tuple[Hashable, Hashable]],
        labels: bool = False,
        label_fill_color: str = BLACK,
        layout: Union[str, dict] = "spring",
        layout_scale: float = 2,
        layout_config: Union[dict, None] = None,
        vertex_type: Type["Mobject"] = Dot,
        vertex_config: Union[dict, None] = None,
        vertex_mobjects: Optional[dict] = None,
        edge_type: Type["Mobject"] = Line,
        partitions: Union[List[List[Hashable]], None] = None,
        root_vertex: Union[Hashable, None] = None,
        edge_config: Union[dict, None] = None,
    ) -> None:
        super().__init__()

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(vertices)
        nx_graph.add_edges_from(edges)
        self._graph = nx_graph

        self._layout = _determine_graph_layout(
            nx_graph,
            layout=layout,
            layout_scale=layout_scale,
            layout_config=layout_config,
            partitions=partitions,
            root_vertex=root_vertex,
        )

        if isinstance(labels, dict):
            self._labels = labels
        elif isinstance(labels, bool):
            if labels:
                self._labels = {
                    v: MathTex(v, fill_color=label_fill_color) for v in vertices
                }
            else:
                self._labels = {}

        if self._labels and vertex_type is Dot:
            vertex_type = LabeledDot

        if vertex_mobjects is None:
            vertex_mobjects = {}

        # build vertex_config
        if vertex_config is None:
            vertex_config = {}
        default_vertex_config = {}
        if vertex_config:
            default_vertex_config = {
                k: v for k, v in vertex_config.items() if k not in vertices
            }
        self._vertex_config = {
            v: vertex_config.get(v, copy(default_vertex_config)) for v in vertices
        }
        self.default_vertex_config = default_vertex_config
        for v, label in self._labels.items():
            self._vertex_config[v]["label"] = label

        self.vertices = {v: vertex_type(**self._vertex_config[v]) for v in vertices}
        self.vertices.update(vertex_mobjects)
        for v in self.vertices:
            self[v].move_to(self._layout[v])

        # build edge_config
        if edge_config is None:
            edge_config = {}
        default_edge_config = {}
        if edge_config:
            default_edge_config = {
                k: v
                for k, v in edge_config.items()
                if k not in edges and k[::-1] not in edges
            }
        self._edge_config = {}
        for e in edges:
            if e in edge_config:
                self._edge_config[e] = edge_config[e]
            elif e[::-1] in edge_config:
                self._edge_config[e] = edge_config[e[::-1]]
            else:
                self._edge_config[e] = copy(default_edge_config)

        self.default_edge_config = default_edge_config
        self.edges = {
            (u, v): edge_type(
                self[u].get_center(),
                self[v].get_center(),
                z_index=-1,
                **self._edge_config[(u, v)],
            )
            for (u, v) in edges
        }

        self.add(*self.vertices.values())
        self.add(*self.edges.values())

        def update_edges(graph):
            for (u, v), edge in graph.edges.items():
                edge.put_start_and_end_on(graph[u].get_center(), graph[v].get_center())

        self.add_updater(update_edges)

    def __getitem__(self: "Graph", v: Hashable) -> "Mobject":
        return self.vertices[v]

    def __repr__(self: "Graph") -> str:
        return f"Graph on {len(self.vertices)} vertices and {len(self.edges)} edges"

    def _add_vertex(
        self,
        vertex: Hashable,
        position: Optional[np.ndarray] = None,
        label: bool = False,
        label_fill_color: str = BLACK,
        vertex_type: Type["Mobject"] = Dot,
        vertex_config: Optional[dict] = None,
        vertex_mobject: Optional[dict] = None,
    ) -> "Mobject":
        """Add a vertex to the graph.

        Parameters
        ----------

        vertex
            A hashable vertex identifier.
        position
            The coordinates where the new vertex should be added. If ``None``, the center
            of the graph is used.
        label
            Controls whether or not the vertex is labeled. If ``False`` (the default),
            the vertex is not labeled; if ``True`` it is labeled using its
            names (as specified in ``vertex``) via :class:`~.MathTex`. Alternatively,
            any :class:`~.Mobject` can be passed to be used as the label.
        label_fill_color
            Sets the fill color of the default labels generated when ``labels``
            is set to ``True``. Has no effect for other values of ``label``.
        vertex_type
            The mobject class used for displaying vertices in the scene.
        vertex_config
            A dictionary containing keyword arguments to be passed to
            the class specified via ``vertex_type``.
        vertex_mobject
            The mobject to be used as the vertex. Overrides all other
            vertex customization options.
        """
        if position is None:
            position = self.get_center()

        if vertex_config is None:
            vertex_config = {}

        if vertex in self.vertices:
            raise ValueError(
                f"Vertex identifier '{vertex}' is already used for a vertex in this graph.",
            )

        self._graph.add_node(vertex)
        self._layout[vertex] = position

        if isinstance(label, (Mobject, OpenGLMobject)):
            self._labels[vertex] = label
        elif label is True:
            self._labels[vertex] = MathTex(vertex, fill_color=label_fill_color)

        base_vertex_config = copy(self.default_vertex_config)
        base_vertex_config.update(vertex_config)
        vertex_config = base_vertex_config

        if vertex in self._labels:
            vertex_config["label"] = self._labels[vertex]
            if vertex_type is Dot:
                vertex_type = LabeledDot

        self._vertex_config[vertex] = vertex_config

        if vertex_mobject is None:
            self.vertices[vertex] = vertex_type(**vertex_config)
        else:
            self.vertices[vertex] = vertex_mobject

        self.vertices[vertex].move_to(position)
        self.add(self.vertices[vertex])

        return self.vertices[vertex]

    def add_vertices(
        self: "Graph",
        *vertices: Hashable,
        positions: Optional[dict] = None,
        labels: bool = False,
        label_fill_color: str = BLACK,
        vertex_type: Type["Mobject"] = Dot,
        vertex_config: Optional[dict] = None,
        vertex_mobjects: Optional[dict] = None,
    ):
        """Add a list of vertices to the graph.

        Parameters
        ----------

        vertices
            Hashable vertex identifiers.
        positions
            A dictionary specifying the coordinates where the new vertices should be added.
            If ``None``, all vertices are created at the center of the graph.
        labels
            Controls whether or not the vertex is labeled. If ``False`` (the default),
            the vertex is not labeled; if ``True`` it is labeled using its
            names (as specified in ``vertex``) via :class:`~.MathTex`. Alternatively,
            any :class:`~.Mobject` can be passed to be used as the label.
        label_fill_color
            Sets the fill color of the default labels generated when ``labels``
            is set to ``True``. Has no effect for other values of ``labels``.
        vertex_type
            The mobject class used for displaying vertices in the scene.
        vertex_config
            A dictionary containing keyword arguments to be passed to
            the class specified via ``vertex_type``.
        vertex_mobjects
            A dictionary whose keys are the vertex identifiers, and whose
            values are mobjects that should be used as vertices. Overrides
            all other vertex customization options.
        """
        if positions is None:
            positions = {}
        if vertex_mobjects is None:
            vertex_mobjects = {}

        graph_center = self.get_center()
        base_positions = {v: graph_center for v in vertices}
        base_positions.update(positions)
        positions = base_positions

        if isinstance(labels, bool):
            labels = {v: labels for v in vertices}
        else:
            assert isinstance(labels, dict)
            base_labels = {v: False for v in vertices}
            base_labels.update(labels)
            labels = base_labels

        if vertex_config is None:
            vertex_config = copy(self.default_vertex_config)

        assert isinstance(vertex_config, dict)
        base_vertex_config = copy(self.default_vertex_config)
        base_vertex_config.update(
            {key: val for key, val in vertex_config.items() if key not in vertices},
        )
        vertex_config = {
            v: (vertex_config[v] if v in vertex_config else copy(base_vertex_config))
            for v in vertices
        }

        return [
            self._add_vertex(
                v,
                position=positions[v],
                label=labels[v],
                label_fill_color=label_fill_color,
                vertex_type=vertex_type,
                vertex_config=vertex_config[v],
                vertex_mobject=vertex_mobjects[v] if v in vertex_mobjects else None,
            )
            for v in vertices
        ]

    @override_animate(add_vertices)
    def _add_vertices_animation(self, *args, anim_args=None, **kwargs):
        if anim_args is None:
            anim_args = {}

        animation = anim_args.pop("animation", Create)

        vertex_mobjects = self.add_vertices(*args, **kwargs)
        return AnimationGroup(*(animation(v, **anim_args) for v in vertex_mobjects))

    def _remove_vertex(self, vertex):
        """Remove a vertex (as well as all incident edges) from the graph.

        Parameters
        ----------

        vertex
            The identifier of a vertex to be removed.

        Returns
        -------

        Group
            A mobject containing all removed objects.

        """
        if vertex not in self.vertices:
            raise ValueError(
                f"The graph does not contain a vertex with identifier '{vertex}'",
            )

        self._graph.remove_node(vertex)
        self._layout.pop(vertex)
        if vertex in self._labels:
            self._labels.pop(vertex)
        self._vertex_config.pop(vertex)

        edge_tuples = [e for e in self.edges if vertex in e]
        for e in edge_tuples:
            self._edge_config.pop(e)
        to_remove = [self.edges.pop(e) for e in edge_tuples]
        to_remove.append(self.vertices.pop(vertex))

        self.remove(*to_remove)
        return self.get_group_class()(*to_remove)

    def remove_vertices(self, *vertices):
        """Remove several vertices from the graph.

        Parameters
        ----------

        vertices
            Vertices to be removed from the graph.

        Examples
        --------
        ::

            >>> G = Graph([1, 2, 3], [(1, 2), (2, 3)])
            >>> removed = G.remove_vertices(2, 3); removed
            VGroup(Line, Line, Dot, Dot)
            >>> G
            Graph on 1 vertices and 0 edges

        """
        mobjects = []
        for v in vertices:
            mobjects.extend(self._remove_vertex(v).submobjects)
        return self.get_group_class()(*mobjects)

    @override_animate(remove_vertices)
    def _remove_vertices_animation(self, *vertices, anim_args=None):
        if anim_args is None:
            anim_args = {}

        animation = anim_args.pop("animation", Uncreate)

        mobjects = self.remove_vertices(*vertices)
        return AnimationGroup(*(animation(mobj, **anim_args) for mobj in mobjects))

    def _add_edge(
        self,
        edge: Tuple[Hashable, Hashable],
        edge_type: Type["Mobject"] = Line,
        edge_config: Optional[dict] = None,
    ):
        """Add a new edge to the graph.

        Parameters
        ----------

        edge
            The edge (as a tuple of vertex identifiers) to be added. If a non-existing
            vertex is passed, a new vertex with default settings will be created. Create
            new vertices yourself beforehand to customize them.
        edge_type
            The mobject class used for displaying edges in the scene.
        edge_config
            A dictionary containing keyword arguments to be passed
            to the class specified via ``edge_type``.

        Returns
        -------
        Group
            A group containing all newly added vertices and edges.

        """
        if edge_config is None:
            edge_config = self.default_edge_config.copy()
        added_mobjects = []
        for v in edge:
            if v not in self.vertices:
                added_mobjects.append(self._add_vertex(v))
        u, v = edge

        self._graph.add_edge(u, v)

        base_edge_config = self.default_edge_config.copy()
        base_edge_config.update(edge_config)
        edge_config = base_edge_config
        self._edge_config[(u, v)] = edge_config

        edge_mobject = edge_type(
            self[u].get_center(), self[v].get_center(), z_index=-1, **edge_config
        )
        self.edges[(u, v)] = edge_mobject

        self.add(edge_mobject)
        added_mobjects.append(edge_mobject)
        return self.get_group_class()(*added_mobjects)

    def add_edges(
        self,
        *edges: Tuple[Hashable, Hashable],
        edge_type: Type["Mobject"] = Line,
        edge_config: Optional[dict] = None,
    ):
        """Add new edges to the graph.

        Parameters
        ----------

        edges
            Edges (as tuples of vertex identifiers) to be added. If a non-existing
            vertex is passed, a new vertex with default settings will be created. Create
            new vertices yourself beforehand to customize them.
        edge_type
            The mobject class used for displaying edges in the scene.
        edge_config
            A dictionary either containing keyword arguments to be passed
            to the class specified via ``edge_type``, or a dictionary
            whose keys are the edge tuples, and whose values are dictionaries
            containing keyword arguments to be passed for the construction
            of the corresponding edge.

        Returns
        -------
        Group
            A group containing all newly added vertices and edges.

        """
        if edge_config is None:
            edge_config = {}
        non_edge_settings = {k: v for (k, v) in edge_config.items() if k not in edges}
        base_edge_config = self.default_edge_config.copy()
        base_edge_config.update(non_edge_settings)
        base_edge_config = {e: base_edge_config.copy() for e in edges}
        for e in edges:
            base_edge_config[e].update(edge_config.get(e, {}))
        edge_config = base_edge_config

        added_mobjects = sum(
            (
                self._add_edge(
                    edge,
                    edge_type=edge_type,
                    edge_config=edge_config[edge],
                ).submobjects
                for edge in edges
            ),
            [],
        )
        return self.get_group_class()(*added_mobjects)

    @override_animate(add_edges)
    def _add_edges_animation(self, *args, anim_args=None, **kwargs):
        if anim_args is None:
            anim_args = {}
        animation = anim_args.pop("animation", Create)

        mobjects = self.add_edges(*args, **kwargs)
        return AnimationGroup(*(animation(mobj, **anim_args) for mobj in mobjects))

    def _remove_edge(self, edge: Tuple[Hashable]):
        """Remove an edge from the graph.

        Parameters
        ----------

        edge
            The edge (i.e., a tuple of vertex identifiers) to be removed from the graph.

        Returns
        -------

        Mobject
            The removed edge.

        """
        if edge not in self.edges:
            edge = edge[::-1]
            if edge not in self.edges:
                raise ValueError(f"The graph does not contain a edge '{edge}'")

        edge_mobject = self.edges.pop(edge)

        self._graph.remove_edge(*edge)
        self._edge_config.pop(edge, None)

        self.remove(edge_mobject)
        return edge_mobject

    def remove_edges(self, *edges: Tuple[Hashable]):
        """Remove several edges from the graph.

        Parameters
        ----------
        edges
            Edges to be removed from the graph.

        Returns
        -------
        Group
            A group containing all removed edges.

        """
        edge_mobjects = [self._remove_edge(edge) for edge in edges]
        return self.get_group_class()(*edge_mobjects)

    @override_animate(remove_edges)
    def _remove_edges_animation(self, *edges, anim_args=None):
        if anim_args is None:
            anim_args = {}

        animation = anim_args.pop("animation", Uncreate)

        mobjects = self.remove_edges(*edges)
        return AnimationGroup(*(animation(mobj, **anim_args) for mobj in mobjects))

    @staticmethod
    def from_networkx(nxgraph: nx.classes.graph.Graph, **kwargs) -> "Graph":
        """Build a :class:`~.Graph` from a given ``networkx`` graph.

        Parameters
        ----------

        nxgraph
            A ``networkx`` graph.
        **kwargs
            Keywords to be passed to the constructor of :class:`~.Graph`.

        Examples
        --------

        .. manim:: ImportNetworkxGraph

            import networkx as nx

            nxgraph = nx.erdos_renyi_graph(14, 0.5)

            class ImportNetworkxGraph(Scene):
                def construct(self):
                    G = Graph.from_networkx(nxgraph, layout="spring", layout_scale=3.5)
                    self.play(Create(G))
                    self.play(*[G[v].animate.move_to(5*RIGHT*np.cos(ind/7 * PI) +
                                                     3*UP*np.sin(ind/7 * PI))
                                for ind, v in enumerate(G.vertices)])
                    self.play(Uncreate(G))

        """
        return Graph(list(nxgraph.nodes), list(nxgraph.edges), **kwargs)

    def change_layout(
        self,
        layout: Union[str, dict] = "spring",
        layout_scale: float = 2,
        layout_config: Union[dict, None] = None,
        partitions: Union[List[List[Hashable]], None] = None,
        root_vertex: Union[Hashable, None] = None,
    ) -> "Graph":
        """Change the layout of this graph.

        See the documentation of :class:`~.Graph` for details about the
        keyword arguments.

        Examples
        --------

        .. manim:: ChangeGraphLayout

            class ChangeGraphLayout(Scene):
                def construct(self):
                    G = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5)],
                              layout={1: [-2, 0, 0], 2: [-1, 0, 0], 3: [0, 0, 0],
                                      4: [1, 0, 0], 5: [2, 0, 0]}
                              )
                    self.play(Create(G))
                    self.play(G.animate.change_layout("circular"))
                    self.wait()
        """
        self._layout = _determine_graph_layout(
            self._graph,
            layout=layout,
            layout_scale=layout_scale,
            layout_config=layout_config,
            partitions=partitions,
            root_vertex=root_vertex,
        )
        for v in self.vertices:
            self[v].move_to(self._layout[v])
        return self
