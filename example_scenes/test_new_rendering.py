from manim import *


class Test(Scene):
    groups_api = True

    @group
    def first_section(self) -> None:
        line = Line()
        line.add_updater(lambda m, dt: m.rotate(PI * dt))
        t = Tex(r"Math! $\sum e^{i\theta}$").add_updater(lambda m: m.next_to(line, UP))
        line.to_edge(LEFT)
        self.add(line, t)
        s = Square()
        t = Tex(
            "Hello, world!", stroke_color=RED, fill_color=BLUE, stroke_width=2
        ).to_edge(RIGHT)
        self.add(t)
        self.play(Create(t), Rotate(s, PI / 2))
        self.wait(1)
        self.play(FadeOut(s))

    @group
    def three_mobjects(self) -> None:
        sq = RegularPolygon(6)
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
        self.play(FadeOut(VGroup(sq, c, st)))

    @group
    def never_run(self) -> None:
        self.play(Write(Text("This should never be run")))


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
