#!/usr/bin/env python


from manim import *

# To watch one of these scenes, run the following:
# python --quality m manim -p example_scenes.py SquareToCircle
#
# Use the flag --quality l for a faster rendering at a lower quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have preview of the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the nth animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1920,1080
# for a 1920x1080 video)


class OpeningManim(Scene):
    def construct(self):
        title = Tex(r"This is some \LaTeX")
        basel = MathTex(r"\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}")
        VGroup(title, basel).arrange(DOWN)
        self.play(
            Write(title),
            FadeIn(basel, shift=DOWN),
        )
        self.wait()

        transform_title = Tex("That was a transform")
        transform_title.to_corner(UP + LEFT)
        self.play(
            Transform(title, transform_title),
            LaggedStart(*(FadeOut(obj, shift=DOWN) for obj in basel)),
        )
        self.wait()

        grid = NumberPlane()
        grid_title = Tex("This is a grid", font_size=72)
        grid_title.move_to(transform_title)

        self.add(grid, grid_title)  # Make sure title is on top of grid
        self.play(
            FadeOut(title),
            FadeIn(grid_title, shift=UP),
            Create(grid, run_time=3, lag_ratio=0.1),
        )
        self.wait()

        grid_transform_title = Tex(
            r"That was a non-linear function \\ applied to the grid",
        )
        grid_transform_title.move_to(grid_title, UL)
        grid.prepare_for_nonlinear_transform()
        self.play(
            grid.animate.apply_function(
                lambda p: p
                + np.array(
                    [
                        np.sin(p[1]),
                        np.sin(p[0]),
                        0,
                    ],
                ),
            ),
            run_time=3,
        )
        self.wait()
        self.play(Transform(grid_title, grid_transform_title))
        self.wait()


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        square.flip(RIGHT)
        square.rotate(-3 * TAU / 8)
        circle.set_fill(PINK, opacity=0.5)

        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))


class WarpSquare(Scene):
    def construct(self):
        square = Square()
        self.play(
            ApplyPointwiseFunction(
                lambda point: complex_to_R3(np.exp(R3_to_complex(point))),
                square,
            ),
        )
        self.wait()


class WriteStuff(Scene):
    def construct(self):
        example_text = Tex("This is a some text", tex_to_color_map={"text": YELLOW})
        example_tex = MathTex(
            "\\sum_{k=1}^\\infty {1 \\over k^2} = {\\pi^2 \\over 6}",
        )
        group = VGroup(example_text, example_tex)
        group.arrange(DOWN)
        group.width = config["frame_width"] - 2 * LARGE_BUFF

        self.play(Write(example_text))
        self.play(Write(example_tex))
        self.wait()


class UpdatersExample(Scene):
    def construct(self):
        decimal = DecimalNumber(
            0,
            show_ellipsis=True,
            num_decimal_places=3,
            include_sign=True,
        )
        square = Square().to_edge(UP)

        decimal.add_updater(lambda d: d.next_to(square, RIGHT))
        decimal.add_updater(lambda d: d.set_value(square.get_center()[1]))
        self.add(square, decimal)
        self.play(
            square.animate.to_edge(DOWN),
            rate_func=there_and_back,
            run_time=5,
        )
        self.wait()


class SpiralInExample(Scene):
    def construct(self):
        logo_green = "#81b29a"
        logo_blue = "#454866"
        logo_red = "#e07a5f"

        font_color = "#ece6e2"

        pi = MathTex(r"\pi").scale(7).set_color(font_color)
        pi.shift(2.25 * LEFT + 1.5 * UP)

        circle = Circle(color=logo_green, fill_opacity=0.7, stroke_width=0).shift(LEFT)
        square = Square(color=logo_blue, fill_opacity=0.8, stroke_width=0).shift(UP)
        triangle = Triangle(color=logo_red, fill_opacity=0.9, stroke_width=0).shift(
            RIGHT
        )
        pentagon = Polygon(
            *[
                [np.cos(2 * np.pi / 5 * i), np.sin(2 * np.pi / 5 * i), 0]
                for i in range(5)
            ],
            color=PURPLE_B,
            fill_opacity=1,
            stroke_width=0
        ).shift(UP + 2 * RIGHT)
        shapes = VGroup(triangle, square, circle, pentagon, pi)
        self.play(SpiralIn(shapes, fade_in_fraction=0.9))
        self.wait()
        self.play(FadeOut(shapes))


Triangle.set_default(stroke_width=20)


class LineJoints(Scene):
    def construct(self):
        t1 = Triangle()
        t2 = Triangle(line_join=LineJointType.ROUND)
        t3 = Triangle(line_join=LineJointType.BEVEL)

        grp = VGroup(t1, t2, t3).arrange(RIGHT)
        grp.set(width=config.frame_width - 1)

        self.add(grp)


# See many more examples at https://docs.manim.community/en/stable/examples.html
