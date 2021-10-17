from manim import *


class Test(Scene):
    def construct(self):
        self.add(Circle())
        self.wait()
        self.next_section()
        self.add(Square())
        self.wait()
