from manim import *


class Test(Scene):
    def construct(self):
        a = Circle()
        b = Square()
        a.become(b)
        self.add(a)
