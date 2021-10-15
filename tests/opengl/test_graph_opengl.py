from manim import Dot, Graph, Line, Text


def test_graph_creation(using_opengl_renderer):
    vertices = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    layout = {1: [0, 0, 0], 2: [1, 1, 0], 3: [1, -1, 0], 4: [-1, 0, 0]}
    G_manual = Graph(vertices=vertices, edges=edges, layout=layout)
    assert len(G_manual.vertices) == 4
    assert len(G_manual.edges) == 4
    G_spring = Graph(vertices=vertices, edges=edges)
    assert len(G_spring.vertices) == 4
    assert len(G_spring.edges) == 4


def test_graph_add_vertices(using_opengl_renderer):
    G = Graph([1, 2, 3], [(1, 2), (2, 3)])
    G.add_vertices(4)
    assert len(G.vertices) == 4
    assert len(G.edges) == 2
    G.add_vertices(5, labels={5: Text("5")})
    assert len(G.vertices) == 5
    assert len(G.edges) == 2
    assert 5 in G._labels
    assert 5 in G._vertex_config
    G.add_vertices(6, 7, 8)
    assert len(G.vertices) == 8
    assert len(G._graph.nodes()) == 8


def test_graph_remove_vertices(using_opengl_renderer):
    G = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5)])
    removed_mobjects = G.remove_vertices(3)
    assert len(removed_mobjects) == 3
    assert len(G.vertices) == 4
    assert len(G.edges) == 2
    assert list(G.vertices.keys()) == [1, 2, 4, 5]
    assert list(G.edges.keys()) == [(1, 2), (4, 5)]
    removed_mobjects = G.remove_vertices(4, 5)
    assert len(removed_mobjects) == 3
    assert len(G.vertices) == 2
    assert len(G.edges) == 1
    assert list(G.vertices.keys()) == [1, 2]
    assert list(G.edges.keys()) == [(1, 2)]


def test_graph_add_edges(using_opengl_renderer):
    G = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3)])
    added_mobjects = G.add_edges((1, 3))
    assert isinstance(added_mobjects.submobjects[0], Line)
    assert len(G.vertices) == 5
    assert len(G.edges) == 3
    assert set(G.vertices.keys()) == {1, 2, 3, 4, 5}
    assert set(G.edges.keys()) == {(1, 2), (2, 3), (1, 3)}

    added_mobjects = G.add_edges((1, 42))
    removed_mobjects = added_mobjects.submobjects
    assert isinstance(removed_mobjects[0], Dot)
    assert isinstance(removed_mobjects[1], Line)

    assert len(G.vertices) == 6
    assert len(G.edges) == 4
    assert set(G.vertices.keys()) == {1, 2, 3, 4, 5, 42}
    assert set(G.edges.keys()) == {(1, 2), (2, 3), (1, 3), (1, 42)}

    added_mobjects = G.add_edges((4, 5), (5, 6), (6, 7))
    assert len(added_mobjects) == 5
    assert len(G.vertices) == 8
    assert len(G.edges) == 7
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


def test_graph_remove_edges(using_opengl_renderer):
    G = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)])
    removed_mobjects = G.remove_edges((1, 2))
    assert isinstance(removed_mobjects.submobjects[0], Line)
    assert len(G.vertices) == 5
    assert len(G.edges) == 4
    assert set(G.edges.keys()) == {(2, 3), (3, 4), (4, 5), (1, 5)}
    assert set(G._graph.edges()) == set(G.edges.keys())

    removed_mobjects = G.remove_edges((2, 3), (3, 4), (4, 5), (5, 1))
    assert len(removed_mobjects) == 4
    assert len(G.vertices) == 5
    assert len(G.edges) == 0
    assert set(G._graph.edges()) == set()
    assert set(G.edges.keys()) == set()
