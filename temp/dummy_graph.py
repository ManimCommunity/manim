from __future__ import annotations

from manim import *


class GraphScene(Scene):
    ANIMATE = True

    def construct(self):
        vertices = [1, 2, 3, 4, 5]
        edges = [(1, 2), (2, 3), (3, 5), (3, 1), (1, 5), (4, 4)]
        layout = {
            1: 2 * UP,
            2: 2 * RIGHT,
            3: 2 * DOWN,
            4: 2 * LEFT + DOWN,
            5: 2 * RIGHT + 2 * UP,
        }
        labels = True  # {1: "1", 2: MathTex("2", stroke_color=BLACK, fill_color=RED), 3: "C", 4: Tex("D", color=BLACK), 5: "E"}
        edge_config = {(4, 4): {"color": RED}}  # None # by default

        # Graph generation

        g = DiGraph(
            vertices=vertices,
            edges=edges,
            labels=labels,
            layout=layout,
            edge_config=edge_config,
            label_fill_color=BLUE,
        )

        # Inspect the generated labels
        for vertex in g.vertices:
            print(f"Vertex {vertex} has position {g.vertices[vertex].get_center()}")

        # Rendering the graphs

        if GraphScene.ANIMATE:
            self.play(Create(g))
            self.wait()

        else:
            self.add(g)


with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = GraphScene()
    scene.render()
