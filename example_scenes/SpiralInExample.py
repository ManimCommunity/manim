from manim import *


class SpiralInExample(Scene):
    def construct(self):
        logo_green = "#81b29a"
        logo_blue = "#454866"
        logo_red = "#e07a5f"

        font_color = "#ece6e2"

        pi = MathTex(r"\pi").scale(7).set_color(font_color)
        pi.shift(2.25 * LEFT + 1.5 * UP)

        circle = Circle(color=logo_green, fill_opacity=1).shift(LEFT)
        square = Square(color=logo_blue, fill_opacity=1).shift(UP)
        triangle = Triangle(color=logo_red, fill_opacity=1).shift(RIGHT)
        pentagon = Polygon(*[[np.cos(2*np.pi/5 * i),np.sin(2*np.pi/5 * i),0] for i in range(5)], color=PURPLE_B).shift(UP+2*RIGHT)
        shapes = VGroup(triangle, square, circle, pentagon, pi)
        self.play(SpiralIn(shapes))
        self.wait()
        self.play(FadeOut(shapes))
