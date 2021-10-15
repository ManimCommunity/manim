from manim import *


class Test(Scene):
    def construct(self):
        circle = Circle()
        dot = Dot([1, 0, 0])
        self.wait()
        self.next_section(SectionType.normal, "test")
        self.add(circle, dot)
        self.wait()
