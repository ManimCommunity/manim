from manim import *


class Test(Scene):
    def construct(self) -> None:
        c = Circle()
        self.play(Create(c))


with tempconfig({"renderer": "opengl", "preview": True, "parallel": False}):
    Test().render()
