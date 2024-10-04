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

        class CustomLabelledDot(LabeledDot):
            def __init__(self, **kwargs):
                print("This is a custom LabelledDot class")
                super().__init__(**kwargs)

        edge_config = {
            (1, 2): {"color": RED},
            (2, 3): {"color": BLUE},
            # You can use a custom class for edge labels as long as it can be initialized with a label keyword and a color keyword
            (3, 5): {"color": GREEN, "label_type": CustomLabelledDot},
            # weights have priority over labels in edge_config
            (3, 1): {"color": YELLOW, "label": "3"},
            (1, 5): {"color": ORANGE, "label_text_color": BLUE},
            (4, 4): {"color": PURPLE, "label_background_color": YELLOW},
        }

        # Graph generation

        g = DiGraph(
            vertices=vertices,
            edges=edges,
            labels=labels,
            weights=weights,
            layout=layout,
            edge_config=edge_config,
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
