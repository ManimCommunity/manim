from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "graph"


@frames_comparison
def test_GraphLoopEdge(scene):
    vertices = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (3, 1), (4, 5), (4, 1), (5, 1), (5, 2), (4, 4)]
    labels = True
    layout = "circular"
    g = Graph(vertices, edges, labels=labels, layout=layout)
    scene.add(g)


@frames_comparison
def test_DiGraphLoopEdge(scene):
    vertices = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 4), (3, 2), (4, 5), (4, 1), (5, 1), (5, 3), (4, 4)]
    labels = True
    layout = "circular"
    g = DiGraph(vertices, edges, labels=labels, layout=layout)
    scene.add(g)


@frames_comparison
def test_WeightedGraph(scene):
    vertices = [1, 2, 3, 4, 5]
    edges = [(1, 3), (2, 3), (3, 4), (4, 2), (4, 5), (5, 1), (5, 3)]
    labels = True
    layout = "circular"
    weights = {
        (1, 3): "4",
        (2, 3): Tex("5", color=RED),
        (3, 4): 0,
        (4, 2): "3",
        (4, 5): 3,
        (5, 1): "2",
        (5, 3): "1",
    }
    g = Graph(
        vertices,
        edges,
        labels=labels,
        weights=weights,
        layout=layout,
    )
    scene.add(g)


@frames_comparison
def test_WeightedDiGraph(scene):
    vertices = [1, 2, 3, 4, 5]
    edges = [(1, 4), (2, 3), (3, 4), (1, 3), (4, 2), (5, 4), (5, 1)]
    labels = True
    layout = "circular"
    weights = {
        (1, 4): "1",
        (2, 3): -1,
        (3, 4): MathTex("2"),
        (1, 3): "2",
        (4, 2): 4,
        (5, 4): 1.5,
        (5, 1): "7/2",
    }
    edge_config = {
        (2, 3): {"label_background_color": WHITE, "label_text_color": BLACK},
        (5, 1): {"label_text_color": YELLOW},
    }
    g = DiGraph(
        vertices,
        edges,
        labels=labels,
        weights=weights,
        layout=layout,
        edge_config=edge_config,
    )
    scene.add(g)


@frames_comparison
def test_GraphWeightedLoopEdge(scene):
    vertices = [1, 2, 3, 4, 5]
    edges = [(1, 2), (1, 5), (2, 4), (3, 2), (4, 5), (4, 1), (3, 3), (5, 3)]
    weights = {
        (1, 2): 1,
        (1, 5): 2,
        (2, 4): 1,
        (3, 2): 4,
        (4, 5): 3,
        (4, 1): 5,
        (3, 3): 1,
        (5, 3): 2,
    }
    labels = True
    layout = "circular"
    g = Graph(vertices, edges, labels=labels, layout=layout, weights=weights)
    scene.add(g)


@frames_comparison
def test_DiGraphWeightedLoopEdge(scene):
    vertices = [1, 2, 3, 4, 5]
    edges = [(1, 2), (1, 5), (2, 4), (3, 2), (4, 5), (4, 3), (3, 3), (5, 3)]
    weights = {
        (1, 2): 1,
        (1, 5): 2,
        (2, 4): 1,
        (3, 2): 4,
        (4, 5): 3,
        (4, 3): 5,
        (3, 3): 1,
        (5, 3): 2,
    }
    labels = True
    layout = "circular"
    g = DiGraph(vertices, edges, labels=labels, layout=layout, weights=weights)
    scene.add(g)
