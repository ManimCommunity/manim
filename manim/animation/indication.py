"""Animations drawing attention to particular mobjects."""

__all__ = [
    "FocusOn",
    "Indicate",
    "Flash",
    "CircleIndicate",
    "ShowPassingFlash",
    "ShowCreationThenDestruction",
    "ShowCreationThenFadeOut",
    "AnimationOnSurroundingRectangle",
    "ShowPassingFlashAround",
    "ShowCreationThenDestructionAround",
    "ShowCreationThenFadeAround",
    "ApplyWave",
    "WiggleOutThenIn",
    "TurnInsideOut",
    # New:
    "Circumscribe",
]


from typing import Type, Union

import numpy as np
from colour import Color

from manim._config import logger

from .. import config
from ..animation.animation import Animation
from ..animation.composition import AnimationGroup, Succession
from ..animation.creation import Create, ShowPartial, Uncreate
from ..animation.fading import FadeIn, FadeOut
from ..animation.movement import Homotopy
from ..animation.transform import Transform
from ..constants import *
from ..mobject.geometry import Circle, Dot, Line, Rectangle
from ..mobject.mobject import Mobject
from ..mobject.shape_matchers import SurroundingRectangle
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.bezier import interpolate
from ..utils.color import GREY, YELLOW
from ..utils.rate_functions import there_and_back, wiggle

if typing.TYPE_CHECKING:
    from ..mobject.mobject import Mobject


class FocusOn(Transform):
    """Draw attention to a position be shrinking a spotlight to it.

    Parameters
    ----------
    focus_point
        The point at which to shrink the spotlight. If it is a :class:`.~Mobject` its center will be used.
    opacity
        The opacity of the spotlight.
    color
        The color of the spotlight.
    run_time
        The duration of the animation.
    kwargs : Any
        Additional arguments to be passed to the :class:`~.Succession` constructor

    Examples
    --------
    .. manim:: UsingFocusOn

        class UsingFocusOn(Scene):
            def construct(self):
                dot = Dot(color=YELLOW).shift(DOWN)
                self.add(Tex("Focusing on the dot below:"), dot)
                self.play(FocusOn(dot))
                self.wait()
    """

    def __init__(
        self,
        focus_point: Union[np.ndarray, Mobject],
        opacity: float = 0.2,
        color: str = GREY,
        run_time: float = 2,
        **kwargs
    ) -> None:
        self.focus_point = focus_point
        self.color = color
        self.opacity = opacity
        remover = True
        # Initialize with blank mobject, while create_target
        # and create_starting_mobject handle the meat
        super().__init__(VMobject(), run_time=run_time, remover=remover, **kwargs)

    def create_target(self) -> "Dot":
        little_dot = Dot(radius=0)
        little_dot.set_fill(self.color, opacity=self.opacity)
        little_dot.add_updater(lambda d: d.move_to(self.focus_point))
        return little_dot

    def create_starting_mobject(self) -> "Dot":
        return Dot(
            radius=config["frame_x_radius"] + config["frame_y_radius"],
            stroke_width=0,
            fill_color=self.color,
            fill_opacity=0,
        )


class Indicate(Transform):
    """Draw attention to a Mobject by temporaly resizing and recoloring it.

    Parameters
    ----------
    mobject
        The mobject to indicate.
    scale_factor
        The factor by which the mobject will be temporally scaled
    color
        The color the mobject temporally takes.
    rate_func
        The function definig the animation progress at every point in time.
    kwargs : Any
        Additional arguments to be passed to the :class:`~.Succession` constructor

    Examples
    --------
    .. manim:: UsingIndicate

        class UsingIndicate(Scene):
            def construct(self):
                self.play(Indicate(Tex("Indicate")))
                self.wait()
    """

    def __init__(
        self,
        mobject: "Mobject",
        scale_factor: float = 1.2,
        color: str = YELLOW,
        rate_func: typing.Callable[
            [float, typing.Optional[float]], np.ndarray
        ] = there_and_back,
        **kwargs
    ) -> None:
        self.color = color
        self.scale_factor = scale_factor
        super().__init__(mobject, rate_func=rate_func, **kwargs)

    def create_target(self) -> "Mobject":
        target = self.mobject.copy()
        target.scale_in_place(self.scale_factor)
        target.set_color(self.color)
        return target


class Flash(AnimationGroup):
    """Draw attention to a position be sending out lines in all directions.

    Parameters
    ----------
    point
        The center of the flash lines. If it is a :class:`.~Mobject` its center will be used.
    line_length
        The length of the flash lines.
    num_lines
        The number of flash lines.
    flash_radius
        The distance from `point` at which the flash lines start.
    line_stroke_width
        The stroke width of the flash lines.
    color
        The color of the flash lines.
    time_width
        The time width used for the flash lines. See :class:`.~ShowPassingFlash` for more details.
    run_time
        The duration of the animation.
    kwargs : Any
        Additional arguments to be passed to the :class:`~.Succession` constructor

    Examples
    --------
    .. manim:: UsingFlash

        class UsingFlash(Scene):
            def construct(self):
                dot = Dot(color=YELLOW).shift(DOWN)
                self.add(Tex("Flash the dot below:"), dot)
                self.play(Flash(dot))
                self.wait()

    .. manim:: FlashOnCircle

        class FlashOnCircle(Scene):
            def construct(self):
                radius = 2
                circle = Circle(radius)
                self.add(circle)
                self.play(Flash(
                    circle, line_length=1,
                    num_lines=30, color=RED,
                    flash_radius=radius+SMALL_BUFF,
                    time_width=0.3, run_time=2,
                    rate_func = rush_from
                ))
    """

    def __init__(
        self,
        point: np.ndarray,
        line_length: float = 0.2,
        num_lines: int = 12,
        flash_radius: float = 0.1,
        line_stroke_width: int = 3,
        color: str = YELLOW,
        time_width: float = 1,
        run_time: float = 1.0,
        **kwargs
    ) -> None:
        self.point = point
        self.color = color
        self.line_length = line_length
        self.num_lines = num_lines
        self.flash_radius = flash_radius
        self.line_stroke_width = line_stroke_width
        self.run_time = run_time
        self.time_width = time_width
        self.animation_config = kwargs

        self.lines = self.create_lines()
        animations = self.create_line_anims()
        super().__init__(*animations, group=self.lines)

    def create_lines(self) -> VGroup:
        # TODO change meaning of flash_radius
        lines = VGroup()
        for angle in np.arange(0, TAU, TAU / self.num_lines):
            line = Line(ORIGIN, self.line_length * RIGHT)
            line.shift((self.flash_radius) * RIGHT)
            line.rotate(angle, about_point=ORIGIN)
            lines.add(line)
        lines.set_color(self.color)
        lines.set_stroke(width=self.line_stroke_width)
        lines.add_updater(lambda l: l.move_to(self.point))
        return lines

    def create_line_anims(self) -> typing.Iterable["ShowPassingFlash"]:
        return [
            ShowPassingFlash(
                line,
                time_width=self.time_width,
                run_time=self.run_time,
                **self.animation_config,
            )
            for line in self.lines
        ]


class CircleIndicate(Indicate):
    """Deprecated. Use :class:`~.Circumscribe` instead."""

    def __init__(
        self,
        mobject: "Mobject",
        circle_config: typing.Dict[str, typing.Any] = {"color": YELLOW},
        rate_func: typing.Callable[
            [float, typing.Optional[float]], np.ndarray
        ] = there_and_back,
        remover: bool = True,
        **kwargs
    ) -> None:
        logger.warning(
            "ShowCreationThenDestructionAround has been deprecated in favor of Circumscribe. Please use Circumscribe instead!"
        )
        self.circle_config = circle_config
        circle = self.get_circle(mobject)
        super().__init__(circle, rate_func=rate_func, remover=remover, **kwargs)

    def get_circle(self, mobject: "Mobject") -> Circle:
        circle = Circle(**self.circle_config)
        circle.add_updater(lambda c: c.surround(mobject))
        return circle

    def interpolate_mobject(self, alpha: float) -> None:
        super().interpolate_mobject(alpha)
        self.mobject.set_stroke(opacity=alpha)


class ShowPassingFlash(ShowPartial):
    """Show only a sliver of the VMobject each frame.

    Parameters
    ----------
    mobject
        The mobject whose stroke is animated.
    time_width
        The length of the sliver relative to the length of the stroke.  

    Examples
    --------
    .. manim:: TimeWidthValues

        class TimeWidthValues(Scene):
            def construct(self):
                p = RegularPolygon(5, color=DARK_GRAY, stroke_width=6).scale(3)
                lbl = VMobject()
                self.add(p, lbl)
                p = p.copy().set_color(BLUE)
                for time_width in [0.2, 0.5, 1, 2]:
                    lbl.become(Tex(r"\\texttt{time\\_width={{%.1f}}}"%time_width))
                    self.play(ShowPassingFlash(
                        p.copy().set_color(BLUE),
                        run_time=2,
                        time_width=time_width
                    ))

    See Also
    --------
    :class:`~.Create`

    """

    def __init__(
        self,
        mobject: "VMobject",
        time_width: float = 0.1,
        **kwargs
    ) -> None:
        self.time_width = time_width
        super().__init__(mobject, remover=True, **kwargs)

    def _get_bounds(self, alpha: float) -> typing.Tuple[float]:
        tw = self.time_width
        upper = interpolate(0, 1 + tw, alpha)
        lower = upper - tw
        upper = min(upper, 1)
        lower = max(lower, 0)
        return (lower, upper)

    def finish(self) -> None:
        super().finish()
        for submob, start in self.get_all_families_zipped():
            submob.pointwise_become_partial(start, 0, 1)


class ShowCreationThenDestruction(ShowPassingFlash):
    def __init__(
        self, mobject: "Mobject", time_width: float = 2.0, run_time: float = 1, **kwargs
    ) -> None:
        super().__init__(mobject, time_width=time_width, run_time=run_time, **kwargs)


class ShowCreationThenFadeOut(Succession):
    def __init__(self, mobject: "Mobject", remover: bool = True, **kwargs) -> None:
        super().__init__(Create(mobject), FadeOut(mobject), remover=remover, **kwargs)


class AnimationOnSurroundingRectangle(AnimationGroup):
    """Deprecated. Use :class:`~.Circumscribe` instead or build an Animation using :class:`~SurroundingRectangle`."""

    def __init__(
        self,
        mobject: "Mobject",
        rect_animation: Animation = Animation,
        surrounding_rectangle_config: typing.Dict[str, typing.Any] = {},
        **kwargs
    ) -> None:
        logger.warning(
            "AnimationOnSurroundingRectangle has been deprecated in favor of Circumscribe. Please use Circumscribe instead!"
        )
        # Callable which takes in a rectangle, and spits out some animation.  Could be
        # some animation class, could be something more
        self.rect_animation = rect_animation
        self.surrounding_rectangle_config = surrounding_rectangle_config
        self.mobject_to_surround = mobject

        rect = self.get_rect()
        rect.add_updater(lambda r: r.move_to(mobject))

        super().__init__(
            self.rect_animation(rect, **kwargs),
        )

    def get_rect(self) -> SurroundingRectangle:
        return SurroundingRectangle(
            self.mobject_to_surround, **self.surrounding_rectangle_config
        )


class ShowPassingFlashAround(AnimationOnSurroundingRectangle):
    """Deprecated. Use :class:`~.Circumscribe` instead."""

    def __init__(
        self, mobject: "Mobject", rect_animation: Animation = ShowPassingFlash, **kwargs
    ) -> None:
        logger.warning(
            "ShowPassingFlashAround has been deprecated in favor of Circumscribe. Please use Circumscribe instead!"
        )
        super().__init__(mobject, rect_animation=rect_animation, **kwargs)


class ShowCreationThenDestructionAround(AnimationOnSurroundingRectangle):
    """Deprecated. Use :class:`~.Circumscribe` instead."""

    def __init__(
        self,
        mobject: "Mobject",
        rect_animation: Animation = ShowCreationThenDestruction,
        **kwargs
    ) -> None:
        logger.warning(
            "ShowCreationThenDestructionAround has been deprecated in favor of Circumscribe. Please use Circumscribe instead!"
        )
        super().__init__(mobject, rect_animation=rect_animation, **kwargs)


class ShowCreationThenFadeAround(AnimationOnSurroundingRectangle):
    """Deprecated. Use :class:`~.Circumscribe` instead."""

    def __init__(
        self,
        mobject: "Mobject",
        rect_animation: Animation = ShowCreationThenFadeOut,
        **kwargs
    ) -> None:
        logger.warning(
            "ShowCreationThenFadeAround has been deprecated in favor of Circumscribe. Please use Circumscribe instead!"
        )
        super().__init__(mobject, rect_animation=rect_animation, **kwargs)


class ApplyWave(Homotopy):
    def __init__(
        self,
        mobject: "Mobject",
        direction: np.ndarray = UP,
        amplitude: float = 0.2,
        run_time: float = 1,
        **kwargs
    ) -> None:
        self.direction = direction
        self.amplitude = amplitude
        left_x = mobject.get_left()[0]
        right_x = mobject.get_right()[0]
        vect = self.amplitude * self.direction

        def homotopy(
            x: float, y: float, z: float, t: float
        ) -> typing.Tuple[float, float, float]:
            alpha = (x - left_x) / (right_x - left_x)
            power = np.exp(2.0 * (alpha - 0.5))
            nudge = there_and_back(t ** power)
            return np.array([x, y, z]) + nudge * vect

        super().__init__(homotopy, mobject, run_time=run_time, **kwargs)


class WiggleOutThenIn(Animation):
    def __init__(
        self,
        mobject: "Mobject",
        scale_value: float = 1.1,
        rotation_angle: float = 0.01 * TAU,
        n_wiggles: int = 6,
        scale_about_point: typing.Optional[np.ndarray] = None,
        rotate_about_point: typing.Optional[np.ndarray] = None,
        run_time: float = 2,
        **kwargs
    ) -> None:
        self.scale_value = scale_value
        self.rotation_angle = rotation_angle
        self.n_wiggles = n_wiggles
        self.scale_about_point = scale_about_point
        self.rotate_about_point = rotate_about_point
        super().__init__(mobject, run_time=run_time, **kwargs)

    def get_scale_about_point(self) -> np.ndarray:
        if self.scale_about_point is None:
            return self.mobject.get_center()

    def get_rotate_about_point(self) -> np.ndarray:
        if self.rotate_about_point is None:
            return self.mobject.get_center()

    def interpolate_submobject(
        self, submobject: "Mobject", starting_submobject: "Mobject", alpha: float
    ) -> None:
        submobject.points[:, :] = starting_submobject.points
        submobject.scale(
            interpolate(1, self.scale_value, there_and_back(alpha)),
            about_point=self.get_scale_about_point(),
        )
        submobject.rotate(
            wiggle(alpha, self.n_wiggles) * self.rotation_angle,
            about_point=self.get_rotate_about_point(),
        )


class TurnInsideOut(Transform):
    def __init__(self, mobject: "Mobject", path_arc: float = TAU / 4, **kwargs) -> None:
        super().__init__(mobject, path_arc=path_arc, **kwargs)

    def create_target(self) -> "Mobject":
        return self.mobject.copy().reverse_points()


class Circumscribe(Succession):
    """Draw a temporary line surrounding the mobject.

    Parameters
    ----------
    mobject
        The mobject to be circumscribed.
    shape
        The shape with which to surrond the given mobject. Should be either "rectangle" or "circle"
    fade_in
        Whether to make the surrounding shape to fade in. It will be drawn otherwise.
    fade_out
        Whether to make the surrounding shape to fade out. It will be undrawn otherwise.
    time_width
        The time_width of the drawing and undrawing. Gets ignored if either `fade_in` or `fade_out` is `True`.
    buff
        The distance between the surrounding shape and the given mobject.
    color
        The color of the surrounding shape.
    run_time
        The duration of the entire animation.
    kwargs : Any
        Additional arguments to be passed to the :class:`~.Succession` constructor

    Examples
    --------

    .. manim:: UsingCircumscribe

        class UsingCircumscribe(Scene):
            def construct(self):
                lbl = Tex(r"Circum-\\\\scribe")
                self.add(lbl)
                self.play(Circumscribe(lbl))
                self.play(Circumscribe(lbl, "circle"))
                self.play(Circumscribe(lbl, fade_out=True))
                self.play(Circumscribe(lbl, time_width=2))
                self.play(Circumscribe(lbl, "circle", True))

    """

    def __init__(
        self,
        mobject: Mobject,
        shape: Union[str, Type] = "rectangle",
        fade_in=False,
        fade_out=False,
        time_width=0.3,
        buff: float = SMALL_BUFF,
        color: Color = YELLOW,
        run_time=1,
        stroke_width=DEFAULT_STROKE_WIDTH,
        **kwargs
    ):
        if shape in ("rectangle", "rectangular", Rectangle):
            frame = SurroundingRectangle(
                mobject, color, buff, stroke_width=stroke_width
            )
        elif shape in ("circle", "circular", Circle):
            frame = Circle(color=color, stroke_width=stroke_width).surround(
                mobject, buffer_factor=1
            )
            radius = frame.width / 2
            frame.scale((radius + buff) / radius)
        else:
            raise ValueError('shape should be either "rectangle" or "circle".')

        if fade_in and fade_out:
            super().__init__(
                FadeIn(frame, run_time=run_time / 2),
                FadeOut(frame, run_time=run_time / 2),
                **kwargs,
            )
        elif fade_in:
            frame.reverse_direction()
            super().__init__(
                FadeIn(frame, run_time=run_time / 2),
                Uncreate(frame, run_time=run_time / 2),
                **kwargs,
            )
        elif fade_out:
            super().__init__(
                Create(frame, run_time=run_time / 2),
                FadeOut(frame, run_time=run_time / 2),
                **kwargs,
            )
        else:
            super().__init__(
                ShowPassingFlash(frame, time_width, run_time=run_time), **kwargs
            )
