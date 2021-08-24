from manim import Graph, Text


def test_graph_creation():
    vertices = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    layout = {1: [0, 0, 0], 2: [1, 1, 0], 3: [1, -1, 0], 4: [-1, 0, 0]}
    G_manual = Graph(vertices=vertices, edges=edges, layout=layout)
    assert str(G_manual) == "Graph on 4 vertices and 4 edges"
    G_spring = Graph(vertices=vertices, edges=edges)
    assert str(G_spring) == "Graph on 4 vertices and 4 edges"


def test_graph_add_vertices():
    G = Graph([1, 2, 3], [(1, 2), (2, 3)])
    G.add_vertices(4)
    assert str(G) == "Graph on 4 vertices and 2 edges"
    G.add_vertices(5, labels={5: Text("5")})
    assert str(G) == "Graph on 5 vertices and 2 edges"
    assert 5 in G._labels
    assert 5 in G._vertex_config
    G.add_vertices(6, 7, 8)
    assert len(G.vertices) == 8
    assert len(G._graph.nodes()) == 8


def test_graph_remove_vertices():
    G = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5)])
    removed_mobjects = G.remove_vertices(3)
    assert len(removed_mobjects) == 3
    assert str(G) == "Graph on 4 vertices and 2 edges"
    assert list(G.vertices.keys()) == [1, 2, 4, 5]
    assert list(G.edges.keys()) == [(1, 2), (4, 5)]
    removed_mobjects = G.remove_vertices(4, 5)
    assert len(removed_mobjects) == 3
    assert str(G) == "Graph on 2 vertices and 1 edges"
    assert list(G.vertices.keys()) == [1, 2]
    assert list(G.edges.keys()) == [(1, 2)]


def test_graph_add_edges():
    G = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3)])
    added_mobjects = G.add_edges((1, 3))
    assert str(added_mobjects.submobjects) == "[Line]"
    assert str(G) == "Graph on 5 vertices and 3 edges"
    assert set(G.vertices.keys()) == {1, 2, 3, 4, 5}
    assert set(G.edges.keys()) == {(1, 2), (2, 3), (1, 3)}

    added_mobjects = G.add_edges((1, 42))
    assert str(added_mobjects.submobjects) == "[Dot, Line]"
    assert str(G) == "Graph on 6 vertices and 4 edges"
    assert set(G.vertices.keys()) == {1, 2, 3, 4, 5, 42}
    assert set(G.edges.keys()) == {(1, 2), (2, 3), (1, 3), (1, 42)}

    added_mobjects = G.add_edges((4, 5), (5, 6), (6, 7))
    assert len(added_mobjects) == 5
    assert str(G) == "Graph on 8 vertices and 7 edges"
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
    assert str(G) == "Graph on 5 vertices and 4 edges"
    assert set(G.edges.keys()) == {(2, 3), (3, 4), (4, 5), (1, 5)}
    assert set(G._graph.edges()) == set(G.edges.keys())

    removed_mobjects = G.remove_edges((2, 3), (3, 4), (4, 5), (5, 1))
    assert len(removed_mobjects) == 4
    assert str(G) == "Graph on 5 vertices and 0 edges"
    assert set(G._graph.edges()) == set()
    assert set(G.edges.keys()) == set()
