"""Mobjects used to mark and annotate other mobjects."""

__all__ = ["SurroundingRectangle", "BackgroundRectangle", "Cross", "Underline"]


from ..constants import *
from ..mobject.geometry import Line, Rectangle
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import BLACK, RED, YELLOW, Color


class SurroundingRectangle(Rectangle):
    r"""A rectangle surrounding a :class:`~.Mobject`

    Examples
    --------

    .. manim:: SurroundingRectExample
        :save_last_frame:

        class SurroundingRectExample(Scene):
            def construct(self):
                title = Title("A Quote from Newton")
                quote = Text(
                    "If I have seen further than others, \n"
                    "it is by standing upon the shoulders of giants.",
                    color=BLUE
                ).scale(0.75)
                box = SurroundingRectangle(quote, color=YELLOW, buff=MED_LARGE_BUFF)
                self.add(title, box, quote)

    """

    def __init__(self, mobject, color=YELLOW, buff=SMALL_BUFF, **kwargs):
        self.color = color
        self.buff = buff
        Rectangle.__init__(
            self,
            color=color,
            width=mobject.width + 2 * self.buff,
            height=mobject.height + 2 * self.buff,
            **kwargs
        )
        self.move_to(mobject)


class BackgroundRectangle(SurroundingRectangle):
    """A background rectangle

    Examples
    --------

    .. manim:: ExampleBackgroundRectangle
        :save_last_frame:

        class ExampleBackgroundRectangle(Scene):
            def construct(self):
                circle = Circle().shift(LEFT)
                circle.set_stroke(color=GREEN, width=20)
                triangle = Triangle().shift(2 * RIGHT)
                triangle.set_fill(PINK, opacity=0.5)
                backgroundRectangle1 = BackgroundRectangle(circle, color=WHITE, fill_opacity=0.15)
                backgroundRectangle2 = BackgroundRectangle(triangle, color=WHITE, fill_opacity=0.15)
                self.add(backgroundRectangle1)
                self.add(backgroundRectangle2)
                self.add(circle)
                self.add(triangle)
                self.play(Rotate(backgroundRectangle1, PI / 4))
                self.play(Rotate(backgroundRectangle2, PI / 2))
    """

    def __init__(
        self,
        mobject,
        color=BLACK,
        stroke_width=0,
        stroke_opacity=0,
        fill_opacity=0.75,
        buff=0,
        **kwargs
    ):
        SurroundingRectangle.__init__(
            self,
            mobject,
            color=color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
            fill_opacity=fill_opacity,
            buff=buff,
            **kwargs
        )
        self.original_fill_opacity = self.fill_opacity

    def pointwise_become_partial(self, mobject, a, b):
        self.set_fill(opacity=b * self.original_fill_opacity)
        return self

    def set_style(
        self,
        stroke_color=None,
        stroke_width=None,
        fill_color=None,
        fill_opacity=None,
        family=True,
    ):
        # Unchangeable style, except for fill_opacity
        super().set_style(
            stroke_color=BLACK,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=fill_opacity,
        )
        return self

    def get_fill_color(self):
        return Color(self.color)


class Cross(VGroup):
    def __init__(self, mobject, stroke_color=RED, stroke_width=6, **kwargs):
        VGroup.__init__(
            self,
            Line(UP + LEFT, DOWN + RIGHT),
            Line(UP + RIGHT, DOWN + LEFT),
        )
        self.replace(mobject, stretch=True)
        self.set_stroke(color=stroke_color, width=stroke_width)


class Underline(Line):
    """Creates an underline.

    Parameters
    ----------
    Line
        The underline.

    Examples
    --------
    .. manim:: UnderLine
        :save_last_frame:

        class UnderLine(Scene):
            def construct(self):
                man = Tex("Manim")  # Full Word
                ul = Underline(man)  # Underlining the word
                self.add(man, ul)
    """

    def __init__(self, mobject, buff=SMALL_BUFF, **kwargs):
        super().__init__(LEFT, RIGHT, buff=buff, **kwargs)
        self.match_width(mobject)
        self.next_to(mobject, DOWN, buff=self.buff)
