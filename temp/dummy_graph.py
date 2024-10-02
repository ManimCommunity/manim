from __future__ import annotations

from manim import *


class GraphScene(Scene):
    ANIMATE = True

    def construct(self):
        vertices = [1, 2, 3, 4, 5]
        edges = [(1, 2), (2, 3), (3, 5), (3, 1), (1, 5), (4, 4)]
        layout = "circular"
        labels = True
        weights = {
            (1, 2): "4",
            (2, 3): Tex("5", color=RED),
            (3, 5): "3",
            (3, 1): 0,
            (1, 5): 3,
            (4, 4): "2",
        }

        # Graph generation

        g = DiGraph(
            vertices=vertices,
            edges=edges,
            labels=labels,
            weights=weights,
            layout=layout,
        )

        # Rendering the graphs

        if GraphScene.ANIMATE:
            self.play(Create(g))
            self.wait()

        else:
            self.add(g)


with tempconfig({"quality": "medium_quality", "preview": True}):
    scene = GraphScene()
    scene.render()
