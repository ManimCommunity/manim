from manim import *


class Test(Scene):
    groups_api = True

    @group
    def first_section(self) -> None:
        line = Line()
        line.add_updater(lambda m, dt: m.rotate(PI * dt))
        tex = Tex(r"Math! $\sum e^{i\theta}$").add_updater(
            lambda m: m.next_to(line, UP)
        )
        line.to_edge(LEFT)
        self.add(line, tex)
        square = Square()
        tex = Tex(
            "Hello, world!", stroke_color=RED, fill_color=BLUE, stroke_width=2
        ).to_edge(RIGHT)
        self.add(tex)
        self.play(Create(tex), Rotate(square, PI / 2))
        self.wait(1)
        self.play(FadeOut(square))

    @group
    def three_mobjects(self) -> None:
        hexagon = RegularPolygon(6)
        circle = Circle()
        star = Star()
        VGroup(hexagon, circle, star).arrange()
        self.play(
            Succession(
                Create(hexagon),
                DrawBorderThenFill(circle),
                SpinInFromNothing(star),
            )
        )
        self.play(FadeOut(VGroup(hexagon, circle, star)))

    @group
    def manim_banner(self) -> None:
        banner = ManimBanner().scale(0.5)
        self.play(banner.create())
        self.play(banner.expand())
        self.wait(1)
        self.play(Unwrite(banner))

    @group
    def graph(self):
        vertices = [1, 2, 3]
        edges = [(1, 2), (2, 3), (3, 1)]
        graph = Graph(vertices, edges, layout="circular")
        self.play(Create(graph))
        self.play(
            graph.animate.add_vertices(
                4,
                5,
                vertex_config={4: {"fill_color": RED}, 5: {"fill_color": RED}},
                positions={4: [2, 1, 0], 5: [2, -1, 0]},
            )
        )
        self.wait(1)
        self.play(Uncreate(graph))


if __name__ == "__main__":
    with (
        tempconfig(
            {
                "preview": True,
                "write_to_movie": False,
                "disable_caching": True,
                "frame_rate": 60,
                "disable_caching_warning": True,
            }
        ),
        Manager(Test) as manager,
    ):
        manager.render()
