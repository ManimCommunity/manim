from manim import *


class Test(Scene):
    def construct(self) -> None:
        s = Square()
        self.add(s)
        self.play(Rotate(s, PI / 2))
        self.play(FadeOut(s))
        sq = Square()
        c = Circle()
        st = Star()
        VGroup(sq, c, st).arrange()
        self.play(
            Succession(
                Create(sq),
                DrawBorderThenFill(c),
                Create(st),
            )
        )


if __name__ == "__main__":
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
