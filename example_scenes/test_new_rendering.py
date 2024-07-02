from manim import *


class Test(Scene):
    def construct(self) -> None:
        s = Square()
        c = Circle()
        st = Star(color=YELLOW, fill_color=YELLOW)
        self.play(
            Succession(*[Create(x) for x in VGroup(s, c, st).arrange()], run_time=2)
        )


with tempconfig(
    {
        "preview": True,
        "write_to_movie": True,
        "disable_caching": True,
        "frame_rate": 60,
        "disable_caching_warning": True,
    }
):
    Manager(Test).render()
