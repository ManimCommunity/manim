from manim import *


class Test(Scene):
    """docstring for Test"""

    def construct(self):
        a = AnnotationDot()
        self.add(a)


class ThreeDTest(ThreeDScene):
    def construct(self):
        a = Dot3D()
        self.add(a)
