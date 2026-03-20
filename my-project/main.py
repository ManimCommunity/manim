from manim import *


class AxesBug(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-6, 6, 2],
            x_axis_config={"unit_size": 1.7},
            y_axis_config={
                # "unit_size": 0.3
            },
        )
        self.add(axes)
