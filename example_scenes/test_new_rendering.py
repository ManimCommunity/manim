from manim import *


class Test(Scene):
    def construct(self) -> None:
        s = Square()
        c = Circle()
        st = Star(color=YELLOW, fill_color=YELLOW)
        self.play(Succession(*[Create(x) for x in VGroup(s, c, st).arrange()]))
        self.wait()
        self.play(*[Uncreate(x) for x in VGroup(s, c, st)])


with tempconfig(
    {
        "write_to_movie": True,
        "disable_caching": True,
    }
):
    Manager(Test).render()
