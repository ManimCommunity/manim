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


# @frames_comparison(last_frame=False)
# def test_digraph_add_edge(scene):
#     vertices = [0, 1]
#     positions = {0: [-1, 0, 0], 1: [1, 0, 0]}
#     g = DiGraph(
#         vertices,
#         [],
#         layout=positions,
#         edge_config={
#             "tip_config": {
#                 "tip_shape": ArrowSquareTip,
#                 "tip_length": 0.15,
#             }
#         },
#     )
#     scene.play(g.animate.add_edges((0, 1)))
#     scene.wait(0.1)


# @frames_comparison(last_frame=False)
# def test_graph_create(scene):
#     graph = Graph(
#         vertices=[1, 2, 3, 4, 5],
#         edges=[
#             (1, 2),
#             (1, 3),
#             (1, 4),
#             (1, 5),
#             (2, 3),
#             (2, 4),
#             (2, 5),
#             (3, 4),
#             (3, 5),
#             (4, 5),
#         ],
#         vertex_type=Circle,
#         vertex_config={"radius": 0.25},
#     )
#     scene.play(
#         AnimationGroup(
#             *(Create(vertex) for vertex in graph.vertices.values()),
#             lag_ratio=0.1,
#         ),
#         run_time=2,
#     )
#     scene.play(
#         AnimationGroup(
#             *(Create(edge) for edge in graph.edges.values()),
#             lag_ratio=0.1,
#         ),
#         run_time=2,
#     )
