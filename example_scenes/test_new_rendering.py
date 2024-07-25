from manim import *


class Test(Scene):
    sections_api = True

    @section
    def first_section(self) -> None:
        s = Square()
        self.add(s)
        self.play(Rotate(s, PI / 2))
        self.wait(7)
        self.play(FadeOut(s))

    @section(skip=True)
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

    @section(skip=True)
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
