from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "graph"


@frames_comparison(last_frame=False)
def test_graph_concurrent_animations(scene):
    vertices = [0, 1]
    positions = {0: [-1, 0, 0], 1: [1, 0, 0]}
    g = Graph(vertices, [], layout=positions)
    scene.play(g[1].animate.move_to([1, 1, 0]), g.animate.add_edges((0, 1)))
    scene.wait(0.1)


@frames_comparison(last_frame=False)
def test_digraph_add_edge(scene):
    vertices = [0, 1]
    positions = {0: [-1, 0, 0], 1: [1, 0, 0]}
    g = DiGraph(
        vertices,
        [],
        layout=positions,
        edge_config={
            "tip_config": {
                "tip_shape": ArrowSquareTip,
                "tip_length": 0.15,
            }
        },
    )
    scene.play(g.animate.add_edges((0, 1)))
    scene.wait(0.1)
