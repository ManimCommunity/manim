from manim import *


class Test(Scene):
    def construct(self) -> None:
        s = Square()
        c = Circle()
        st = Star(color=YELLOW, fill_color=YELLOW)
        self.play(Succession(*[Create(x) for x in VGroup(s, c, st).arrange()]))


with tempconfig({"renderer": "opengl", "preview": True, "parallel": False}):
    Manager(Test).render()
