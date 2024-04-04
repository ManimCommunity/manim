import os

from manim import *


class Test(LinearTransformationScene):
    def construct(self):
        matrix = [[3, 0], [1, -1]]

        self.add_unit_square()
        c = Circle(radius=0.5)
        self.add_moving_mobject(c)
        self.apply_matrix(matrix)
        self.wait()
        self.apply_inverse(matrix)


class Test2(LinearTransformationScene):
    def construct(self):
        t = Text("Hello World!")
        self.play(Write(t))
        self.wait()
        self.play(Unwrite(t))


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": True}):
        Test().render()
