from __future__ import annotations

import pytest

from manim import DiGraph, Graph, LabeledLine, Scene, Text, tempconfig
from manim.mobject.graph import _layouts


def test_graph_creation():
    vertices = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    layout = {1: [0, 0, 0], 2: [1, 1, 0], 3: [1, -1, 0], 4: [-1, 0, 0]}
    G_manual = Graph(vertices=vertices, edges=edges, layout=layout)
    assert str(G_manual) == "Undirected graph on 4 vertices and 4 edges"
    G_spring = Graph(vertices=vertices, edges=edges)
    assert str(G_spring) == "Undirected graph on 4 vertices and 4 edges"
    G_directed = DiGraph(vertices=vertices, edges=edges)
    assert str(G_directed) == "Directed graph on 4 vertices and 4 edges"


def test_graph_add_vertices():
    G = Graph([1, 2, 3], [(1, 2), (2, 3)])
    G.add_vertices(4)
    assert str(G) == "Undirected graph on 4 vertices and 2 edges"
    G.add_vertices(5, labels={5: Text("5")})
    assert str(G) == "Undirected graph on 5 vertices and 2 edges"
    assert 5 in G._labels
    assert 5 in G._vertex_config
    G.add_vertices(6, 7, 8)
    assert len(G.vertices) == 8
    assert len(G._graph.nodes()) == 8


def test_graph_remove_vertices():
    G = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5)])
    removed_mobjects = G.remove_vertices(3)
    assert len(removed_mobjects) == 3
    assert str(G) == "Undirected graph on 4 vertices and 2 edges"
    assert list(G.vertices.keys()) == [1, 2, 4, 5]
    assert list(G.edges.keys()) == [(1, 2), (4, 5)]
    removed_mobjects = G.remove_vertices(4, 5)
    assert len(removed_mobjects) == 3
    assert str(G) == "Undirected graph on 2 vertices and 1 edges"
    assert list(G.vertices.keys()) == [1, 2]
    assert list(G.edges.keys()) == [(1, 2)]


def test_graph_add_edges():
    G = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3)])
    added_mobjects = G.add_edges((1, 3))
    assert str(added_mobjects.submobjects) == "[Line]"
    assert str(G) == "Undirected graph on 5 vertices and 3 edges"
    assert set(G.vertices.keys()) == {1, 2, 3, 4, 5}
    assert set(G.edges.keys()) == {(1, 2), (2, 3), (1, 3)}

    added_mobjects = G.add_edges((1, 42))
    assert str(added_mobjects.submobjects) == "[Dot, Line]"
    assert str(G) == "Undirected graph on 6 vertices and 4 edges"
    assert set(G.vertices.keys()) == {1, 2, 3, 4, 5, 42}
    assert set(G.edges.keys()) == {(1, 2), (2, 3), (1, 3), (1, 42)}

    added_mobjects = G.add_edges((4, 5), (5, 6), (6, 7))
    assert len(added_mobjects) == 5
    assert str(G) == "Undirected graph on 8 vertices and 7 edges"
    assert set(G.vertices.keys()) == {1, 2, 3, 4, 5, 42, 6, 7}
    assert set(G._graph.nodes()) == set(G.vertices.keys())
    assert set(G.edges.keys()) == {
        (1, 2),
        (2, 3),
        (1, 3),
        (1, 42),
        (4, 5),
        (5, 6),
        (6, 7),
    }
    assert set(G._graph.edges()) == set(G.edges.keys())


def test_graph_remove_edges():
    G = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)])
    removed_mobjects = G.remove_edges((1, 2))
    assert str(removed_mobjects.submobjects) == "[Line]"
    assert str(G) == "Undirected graph on 5 vertices and 4 edges"
    assert set(G.edges.keys()) == {(2, 3), (3, 4), (4, 5), (1, 5)}
    assert set(G._graph.edges()) == set(G.edges.keys())

    removed_mobjects = G.remove_edges((2, 3), (3, 4), (4, 5), (1, 5))
    assert len(removed_mobjects) == 4
    assert str(G) == "Undirected graph on 5 vertices and 0 edges"
    assert set(G._graph.edges()) == set()
    assert set(G.edges.keys()) == set()


def test_graph_accepts_labeledline_as_edge_type():
    vertices = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    edge_config = {
        (1, 2): {"label": "A"},
        (2, 3): {"label": "B"},
        (3, 4): {"label": "C"},
        (4, 1): {"label": "D"},
    }
    G_manual = Graph(vertices, edges, edge_type=LabeledLine, edge_config=edge_config)
    G_directed = DiGraph(
        vertices, edges, edge_type=LabeledLine, edge_config=edge_config
    )

    for edge_obj in G_manual.edges.values():
        assert isinstance(edge_obj, LabeledLine)
        assert hasattr(edge_obj, "label")

    for edge_obj in G_directed.edges.values():
        assert isinstance(edge_obj, LabeledLine)
        assert hasattr(edge_obj, "label")


def test_custom_animation_mobject_list():
    G = Graph([1, 2, 3], [(1, 2), (2, 3)])
    scene = Scene()
    scene.add(G)
    assert scene.mobjects == [G]
    with tempconfig({"dry_run": True, "quality": "low_quality"}):
        scene.play(G.animate.add_vertices(4))
        assert str(G) == "Undirected graph on 4 vertices and 2 edges"
        assert scene.mobjects == [G]
        scene.play(G.animate.remove_vertices(2))
        assert str(G) == "Undirected graph on 3 vertices and 0 edges"
        assert scene.mobjects == [G]


def test_custom_graph_layout_dict():
    G = Graph(
        [1, 2, 3], [(1, 2), (2, 3)], layout={1: [0, 0, 0], 2: [1, 1, 0], 3: [1, -1, 0]}
    )
    assert str(G) == "Undirected graph on 3 vertices and 2 edges"
    assert all(G.vertices[1].get_center() == [0, 0, 0])
    assert all(G.vertices[2].get_center() == [1, 1, 0])
    assert all(G.vertices[3].get_center() == [1, -1, 0])


def test_graph_layouts():
    for layout in (layout for layout in _layouts if layout not in ["tree", "partite"]):
        G = Graph([1, 2, 3], [(1, 2), (2, 3)], layout=layout)
        assert str(G) == "Undirected graph on 3 vertices and 2 edges"


def test_tree_layout():
    G = Graph([1, 2, 3], [(1, 2), (2, 3)], layout="tree", root_vertex=1)
    assert str(G) == "Undirected graph on 3 vertices and 2 edges"


def test_partite_layout():
    G = Graph(
        [1, 2, 3, 4, 5],
        [(1, 2), (2, 3), (3, 4), (4, 5)],
        layout="partite",
        partitions=[[1, 2], [3, 4, 5]],
    )
    assert str(G) == "Undirected graph on 5 vertices and 4 edges"


def test_custom_graph_layout_function():
    def layout_func(graph, scale):
        return {vertex: [vertex, vertex, 0] for vertex in graph}

    G = Graph([1, 2, 3], [(1, 2), (2, 3)], layout=layout_func)
    assert all(G.vertices[1].get_center() == [1, 1, 0])
    assert all(G.vertices[2].get_center() == [2, 2, 0])
    assert all(G.vertices[3].get_center() == [3, 3, 0])


def test_custom_graph_layout_function_with_kwargs():
    def layout_func(graph, scale, offset):
        return {
            vertex: [vertex * scale + offset, vertex * scale + offset, 0]
            for vertex in graph
        }

    G = Graph(
        [1, 2, 3], [(1, 2), (2, 3)], layout=layout_func, layout_config={"offset": 1}
    )
    assert all(G.vertices[1].get_center() == [3, 3, 0])
    assert all(G.vertices[2].get_center() == [5, 5, 0])
    assert all(G.vertices[3].get_center() == [7, 7, 0])


def test_graph_change_layout():
    for layout in (layout for layout in _layouts if layout not in ["tree", "partite"]):
        G = Graph([1, 2, 3], [(1, 2), (2, 3)])
        G.change_layout(layout=layout)
        assert str(G) == "Undirected graph on 3 vertices and 2 edges"


def test_tree_layout_no_root_error():
    with pytest.raises(ValueError) as excinfo:
        G = Graph([1, 2, 3], [(1, 2), (2, 3)], layout="tree")
    assert str(excinfo.value) == "The tree layout requires the root_vertex parameter"


def test_tree_layout_not_tree_error():
    with pytest.raises(ValueError) as excinfo:
        G = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], layout="tree", root_vertex=1)
    assert str(excinfo.value) == "The tree layout must be used with trees"
