from manim import *


class Test(Scene):
    def construct(self) -> None:
        b = ManimBanner()
        # self.play(b.expand())
        self.play(DrawBorderThenFill(b))


with tempconfig({"renderer": "opengl", "preview": True, "parallel": False}):
    Test().render()
