from manim import *


class Test(Scene):
    def construct(self) -> None:
        s = Square()
        self.add(s)
        self.play(Rotate(s, PI / 2))
        self.play(FadeOut(s))
        sq = RegularPolygon(6)
        c = Circle()
        st = Star()
        spinny = Line().to_edge(LEFT)
        spinny.add_dt_updater(lambda m, dt: m.rotate(PI / 2 * dt))
        txt = Text("Spinny")
        txt.add_updater(lambda m: m.next_to(spinny, DOWN))
        self.add(spinny, txt)
        VGroup(sq, c, st).arrange()
        self.play(
            Succession(
                Create(sq),
                DrawBorderThenFill(c),
                Create(st),
            )
        )
        self.play(FadeOut(VGroup(*self.mobjects)))


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
