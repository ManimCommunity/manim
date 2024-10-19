"""Mobjects used to mark and annotate other mobjects."""

from __future__ import annotations

__all__ = ["SurroundingRectangle", "BackgroundRectangle", "Cross", "Underline"]

from typing import Any

from typing_extensions import Self

from manim import logger
from manim._config import config
from manim.constants import (
    DOWN,
    LEFT,
    RIGHT,
    SMALL_BUFF,
    UP,
)
from manim.mobject.geometry.line import Line
from manim.mobject.geometry.polygram import RoundedRectangle
from manim.mobject.mobject import Mobject
from manim.mobject.types.vectorized_mobject import VGroup
from manim.utils.color import BLACK, RED, YELLOW, ManimColor, ParsableManimColor


class SurroundingRectangle(RoundedRectangle):
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
                    color=BLUE,
                ).scale(0.75)
                box = SurroundingRectangle(quote, color=YELLOW, buff=MED_LARGE_BUFF)

                t2 = Tex(r"Hello World").scale(1.5)
                box2 = SurroundingRectangle(t2, corner_radius=0.2)
                mobjects = VGroup(VGroup(box, quote), VGroup(t2, box2)).arrange(DOWN)
                self.add(title, mobjects)
    """

    def __init__(
        self,
        *mobjects: Mobject,
        color: ParsableManimColor = YELLOW,
        buff: float = SMALL_BUFF,
        corner_radius: float = 0.0,
        **kwargs: Any,
    ) -> None:
        from manim.mobject.mobject import Group

        if not all(isinstance(mob, Mobject) for mob in mobjects):
            raise TypeError(
                "Expected all inputs for parameter mobjects to be a Mobjects"
            )

        group = Group(*mobjects)
        super().__init__(
            color=color,
            width=group.width + 2 * buff,
            height=group.height + 2 * buff,
            corner_radius=corner_radius,
            **kwargs,
        )
        self.buff = buff
        self.move_to(group)


class BackgroundRectangle(SurroundingRectangle):
    """A background rectangle. Its default color is the background color
    of the scene.

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
        *mobjects: Mobject,
        color: ParsableManimColor | None = None,
        stroke_width: float = 0,
        stroke_opacity: float = 0,
        fill_opacity: float = 0.75,
        buff: float = 0,
        **kwargs: Any,
    ) -> None:
        if color is None:
            color = config.background_color

        super().__init__(
            *mobjects,
            color=color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
            fill_opacity=fill_opacity,
            buff=buff,
            **kwargs,
        )
        self.original_fill_opacity: float = self.fill_opacity

    def pointwise_become_partial(self, mobject: Mobject, a: Any, b: float) -> Self:
        self.set_fill(opacity=b * self.original_fill_opacity)
        return self

    def set_style(self, fill_opacity: float, **kwargs: Any) -> Self:  # type: ignore[override]
        # Unchangeable style, except for fill_opacity
        # All other style arguments are ignored
        super().set_style(
            stroke_color=BLACK,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=fill_opacity,
        )
        if len(kwargs) > 0:
            logger.info(
                "Argument %s is ignored in BackgroundRectangle.set_style.",
                kwargs,
            )
        return self

    def get_fill_color(self) -> ManimColor:
        # The type of the color property is set to Any using the property decorator
        # vectorized_mobject.py#L571
        temp_color: ManimColor = self.color
        return temp_color


class Cross(VGroup):
    """Creates a cross.

    Parameters
    ----------
    mobject
        The mobject linked to this instance. It fits the mobject when specified. Defaults to None.
    stroke_color
        Specifies the color of the cross lines. Defaults to RED.
    stroke_width
        Specifies the width of the cross lines. Defaults to 6.
    scale_factor
        Scales the cross to the provided units. Defaults to 1.

    Examples
    --------
    .. manim:: ExampleCross
        :save_last_frame:

        class ExampleCross(Scene):
            def construct(self):
                cross = Cross()
                self.add(cross)
    """

    def __init__(
        self,
        mobject: Mobject | None = None,
        stroke_color: ParsableManimColor = RED,
        stroke_width: float = 6.0,
        scale_factor: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            Line(UP + LEFT, DOWN + RIGHT), Line(UP + RIGHT, DOWN + LEFT), **kwargs
        )
        if mobject is not None:
            self.replace(mobject, stretch=True)
        self.scale(scale_factor)
        self.set_stroke(color=stroke_color, width=stroke_width)


class Underline(Line):
    """Creates an underline.

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

    def __init__(
        self, mobject: Mobject, buff: float = SMALL_BUFF, **kwargs: Any
    ) -> None:
        super().__init__(LEFT, RIGHT, buff=buff, **kwargs)
        self.match_width(mobject)
        self.next_to(mobject, DOWN, buff=self.buff)
