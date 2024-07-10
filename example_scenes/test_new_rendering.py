from manim import *


class Test(Scene):
    def construct(self) -> None:
        s = Square()
        self.add(s)
        self.play(Rotate(s, PI / 2))


with tempconfig(
    {
        "preview": True,
        "write_to_movie": False,
        "disable_caching": True,
        "frame_rate": 60,
        "disable_caching_warning": True,
    }
):
    Manager(Test).render()
