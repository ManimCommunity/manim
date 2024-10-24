from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "graph"


@frames_comparison
def test_digraph_add_edges(scene):
    vertices = range(5)
    edges = [
        (0, 1),
        (1, 2),
        (3, 2),
        (3, 4),
    ]

    edge_config = {
        "stroke_width": 2,
        "tip_config": {
            "tip_shape": ArrowSquareTip,
            "tip_length": 0.15,
        },
        (3, 4): {"color": RED, "tip_config": {"tip_length": 0.25, "tip_width": 0.25}},
    }

    g = DiGraph(
        vertices,
        [],
        labels=True,
        layout="circular",
    ).scale(1.4)

    g.add_edges(*edges, edge_config=edge_config)

    scene.add(g)
