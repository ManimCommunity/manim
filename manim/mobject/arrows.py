__all__ = ["ArrowTip", "Arrow", "Vector"]

from functools import cached_property, wraps
from typing import Literal, Optional, Union

import numpy as np
from colour import Color

from .. import config
from ..constants import *
from ..utils.space_ops import angle_of_vector
from .geometry import Line, Triangle
from .matrix import Matrix
from .mobject import Mobject
from .opengl_mobject import OpenGLMobject
from .types.opengl_vectorized_mobject import OpenGLVMobject
from .types.vectorized_mobject import MetaVMobject, VMobject

DEFAULT_ARROW_TO_STROKE_WIDTH_RATIO = 35 / 6
# TODO needs cleanup


import types

# TODO: add tip presets (via string?)


class ArrowTip:
    """An arrow tip.

    An arrow tip is not a :class:`~.Mobject` but always contains one in the
    attribute ``mobject``. This mobject is placed on the stroke of a
    :class:`~.VMobject`.

    Parameters
    ----------
    base_line
        The vmobject on whose stroke the tip is placed.
    mobject
        The mobject placed on the stroke of the given ``base_line``. If ``mobject``
        is ``None``, a new :class:`~.Triangle` is used as tip.
    relative_position
        Where on the `base_line` the tip should be placed. ``0`` is the start of the
        line and ``1`` is its end.
    tip_angle
        The angle in which the tip mobject is pointing before adding it. By default
        it is assumed to be pointing up.
    backwards
        Whether the arrow tip should be pointing against the direction of the
        base_line. Effectively this will invert ``relative_position``, ``tip_angle``
        and ``tip_alignment``.
    tip_alignment
        The side of the tip to be positioned at the specified ``relative_position``
        assuming the arrow points to the right.
    scale_auto
        Whether to automatically scale the given ``mobject``.  #TODO
    length
        The length of the tip.
    width
        The width of the tip.
    color
        The color of the tip. If ``color`` is ``None`` the color of the given
        ``mobject`` is not modified. If ``color == "copy"`` the stroke_color of the
        ``base_line`` is used.
    filled
        Whether the tip should be filled. If ``filled`` is ``None`` the fill of the
        given ``mobject`` is not modified.
    secant_delta
        The difference in ``relative_position`` to be used to calculate the angle of
        the ``base_line`` at the required position. If ``secant_delta`` is ``None``
        a fitting value is chosen based on the length of the tip and ``base_line``.

    Examples
    --------

    .. manim:: ArrowTips
        :save_last_frame:

        class ArrowTips(Scene):
            def construct(self):
                arcs = VGroup(
                    *[
                        Line(LEFT * 4, RIGHT * 4, path_arc=arc, stroke_width=6)
                        for arc in np.linspace(1, -1, 6)
                    ]
                ).arrange_submobjects(DOWN, buff=0.7)
                self.add(arcs)

                arcs[0].add_tip()
                arcs[1].add_tip(Square(), tip_angle=PI / 4, length=0.4, width=0.25)
                arcs[2].add_tip(color=RED, filled=False, backwards=True)
                arcs[3].add_tip(tip_alignment=ORIGIN, filled=False)
                arcs[4].add_tip(tip_alignment=RIGHT, filled=False)

                for pos in [0.2, 0.4, 0.6, 0.8]:
                    arcs[5].add_tip(relative_position=pos, tip_alignment=ORIGIN)

    .. manim:: CornerExample

        class CornerExample(Scene):
            def construct(self):
                elbow = Elbow(4, angle=PI / 4, stroke_width=30).move_to(ORIGIN)
                self.add(elbow)
                elbow.add_tip(tip_alignment=ORIGIN, color=RED)
                #secant_delta and tip_alignment influence how the arrow behaves close to corners

                def updater(elbow, alpha):
                    alpha = alpha * 0.5 + 0.25
                    elbow.tips[0].set_relative_position(alpha)

                self.play(UpdateFromAlphaFunc(elbow, updater, run_time=1.5, rate_func=linear))
    """

    def __new__(
        cls,
        base_line: MetaVMobject,
        mobject: Optional[Union[Mobject, OpenGLMobject]] = None,
        *,
        relative_position: float = 1,
        tip_angle: float = PI / 2,
        backwards: bool = False,
        tip_alignment=LEFT,  # or RIGHT or ORIGIN
        scale_auto=True,
        length: Optional[float] = None,
        width: Optional[float] = None,
        color: Optional[Union[Color, Literal["copy"]]] = None,
        filled: Optional[bool] = None,
        secant_delta: Optional[float] = None,
        **kwargs,
    ):
        if mobject is None:
            mobject = Triangle()
            mobject.width = DEFAULT_ARROW_TIP_LENGTH
            mobject.stretch_to_fit_height(DEFAULT_ARROW_TIP_LENGTH)
            filled = filled is None or filled
            color = "copy" if color is None else color
        elif hasattr(mobject, "tip_attrs"):  # mobject is used as tip elsewhere
            cls.trim(mobject)

        cls.extend(mobject)

        if backwards:
            relative_position = 1 - relative_position
            tip_angle += PI
            tip_alignment = tip_alignment * -1

        tip_attrs = {
            "base_line": base_line,
            "relative_position": relative_position,
            "secant_delta": 1e-6 if secant_delta == 0 else secant_delta,
            "tip_angle": tip_angle,
            "tip_alignment": tip_alignment,
        }
        mobject.tip_attrs = tip_attrs

        # ignore scale_auto if length and width are defined
        if length is not None and width is not None:
            mobject.set_tip_length(length, proportional=False, update=False)
            mobject.set_tip_width(width, proportional=False, update=False)
        # use scale_auto to decide if scaling is proportional if only one dim is defined.
        elif length is not None:
            mobject.set_tip_length(length, proportional=scale_auto, update=False)
        elif width is not None:
            mobject.set_tip_width(length, proportional=scale_auto, update=False)
        # choose width depending on base line stroke width
        elif scale_auto:
            width = base_line.stroke_width * DEFAULT_ARROW_TO_STROKE_WIDTH_RATIO / 100
            mobject.set_tip_width(width, update=False)

        if color:
            if color == "copy":
                color = base_line.get_stroke_color()
            mobject.set_color(color)
        if filled is not None and isinstance(mobject, (VMobject, OpenGLVMobject)):
            mobject.set_fill(opacity=float(filled))

        mobject.update_tip_positioning()
        return mobject

    @classmethod
    def _methods_list(cls):
        return filter(
            lambda name: not (
                name.startswith("__")
                and name.endswith("__")
                or type(cls.__dict__[name]) != types.FunctionType
            ),
            cls.__dict__,
        )

    @classmethod
    def extend(cls, mobject: Mobject):
        """Add this class as mixin to a mobject.

        Parameters
        ----------
        mobject
            The mobject to be extended.
        """

        for name in cls._methods_list():
            if name in mobject.__dict__:
                raise ValueError()
            mobject.__dict__[name] = cls.__dict__[name].__get__(mobject)

    @classmethod
    def trim(cls, tip: "ArrowTip"):
        tip.rotate(-tip.tip_attrs["tip_angle"] + PI / 2)  # add pi/2 to rotate up
        for name in cls._methods_list():
            tip.__dict__.pop(name)
        delattr(tip, "tip_attrs")

    def _unrotated_tip(func):
        @wraps(func)
        def method_wrapper(self, *args, update=True, **kwargs):
            if self.tip_attrs["tip_angle"] != 0:
                self.rotate(-self.tip_attrs["tip_angle"])
                self.tip_attrs["tip_angle"] = 0
            result = func(self, *args, **kwargs)
            if update:
                self.update_tip_positioning()
            return result

        return method_wrapper

    @_unrotated_tip
    def set_tip_length(self, length, proportional=True):
        """Set the length of the tip.

        Parameters
        ----------
        length
            The new length of the arrow tip.
        proportional
            Whether to scale the width of the tip proportionally.
        """
        if proportional:
            self.width = length
        else:
            self.stretch_to_fit_width(length)

    @_unrotated_tip
    def get_tip_length(self) -> float:
        """Get the length of the arrow tip.

        Returns
        -------
        float
            The length of the arrow tip.
        """
        return self.width

    @_unrotated_tip
    def set_tip_width(self, width, proportional=True):
        """Set the width of the tip.

        Parameters
        ----------
        width
            The new width of the arrow tip.
        proportional
            Whether to scale the length of the tip proportionally.
        """
        if proportional:
            self.height = width
        else:
            self.stretch_to_fit_height(width)

    @_unrotated_tip
    def get_tip_width(self):
        """Get the width of the arrow tip.

        Returns
        -------
        float
            The width of the arrow tip.
        """
        return self.height

    def set_relative_tip_position(self, relative_position):
        self.tip_attrs["relative_position"] = relative_position
        self.update_tip_positioning()

    def update_tip_positioning(self):
        """Update the positioning of the tip."""
        t = self.tip_attrs
        if t["tip_angle"]:
            self.rotate(-t["tip_angle"])
        position = t["base_line"].point_from_proportion(t["relative_position"])
        self.move_to(position, t["tip_alignment"])
        rotation_about_point = self.get_critical_point(t["tip_alignment"])

        # calculate tip angle
        delta = t["secant_delta"] or self.width / t["base_line"].get_arc_length() / 2
        pos = t["relative_position"]
        if t["tip_alignment"][0] == 0:  # centered alignment
            p = [pos - delta, pos + delta]
        elif t["tip_alignment"][0] < 0:  # LEFT alignment
            p = [pos - (1e-6 if pos == 1 else 0), pos + delta]
        else:  # RIGHT alignment
            p = [pos - delta, pos + (1e-6 if pos == 0 else 0)]
        p = [*map(lambda x: t["base_line"].point_from_proportion(np.clip(x, 0, 1)), p)]
        t["tip_angle"] = angle_of_vector(p[1] - p[0])

        self.rotate(t["tip_angle"], about_point=rotation_about_point)


class Arrow(Line):
    """An Arrow.

    Parameters
    ----------
    start
        The start point of the arrow.
    end
        The end point of the arrow.
    buff
        The distance between the arrow and the defined start and end position.
    path_arc
        The curvature of the arrow.
    target_stroke_width
        The desired maximum width of the arrow. The stroke of short arrows is scaled
        down automatically based on ``max_stroke_width_to_length_ratio``.
    tip_mobject
        The :class:`~.Mobject` used as tip. See :class:`~.ArrowTip` for more details.
    double
        Whether the tip should be added on both ends.
    max_tip_length_to_length_ratio
        The maximum tip to arrow length ratio. To large arrow tips get scaled
        automatically.
    max_stroke_width_to_length_ratio
        The maximum stroke width to length ratio. To thick arrows are thinned out
        automatically.
    kwargs
        Additional keyword arguments passed to :class:`ArrowTip`

    Examples
    --------
    .. manim:: ArrowTypes
        :save_last_frame:

        class ArrowTypes(Scene):
            def construct(self):
                # Add dots
                dots = VGroup(Dot(3 * UP), Dot(3 * DOWN), Dot([1, 3, 0]), Dot([1, -3, 0]))
                dots.align_on_border(LEFT).set_color(RED)
                self.add(dots)

                # Arrows between points
                self.add(Arrow(dots[0], dots[1]))
                self.add(Arrow(dots[3], dots[2], buff=0))

                # More arrow options
                pos = [LEFT * 3, RIGHT * 4]
                more_arrows = VGroup(
                    Arrow(*pos, path_arc=-PI / 4),
                    Arrow(*pos, double=True, color=BLUE_B),
                    Arrow(*pos, tip_mobject=Circle(), filled=True),  # Tip color is kept
                    Arrow(*pos, backwards=True, color=BLUE_D),
                    Arrow(*pos, path_arc=PI / 4, relative_position=0.5, tip_alignment=ORIGIN),
                ).arrange_submobjects(DOWN, buff=1)
                self.add(more_arrows)

    """

    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        *,
        buff=MED_SMALL_BUFF,
        path_arc=None,
        target_stroke_width=6,
        tip_mobject=None,
        double=False,
        max_tip_length_to_length_ratio=0.2,
        max_stroke_width_to_length_ratio=6,
        **kwargs,
    ):
        super().__init__(start, end, buff, path_arc, stroke_width=target_stroke_width)
        if "color" in kwargs:
            self.set_color(kwargs["color"])
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio
        self.target_stroke_width = target_stroke_width

        kwargs.setdefault("tip_alignment", RIGHT)
        if double:
            self.add_tip(
                None if tip_mobject is None else tip_mobject.copy(),
                backwards=True,
                **kwargs,
            )
        self.add_tip(tip_mobject, **kwargs)
        tip = self.get_tip(-1)
        self.target_tip_length = tip.get_tip_length()

        self.update_stroke_and_tips()

    def update_stroke_and_tips(self):
        max_from_ratio = self.max_stroke_width_to_length_ratio * self.get_arc_length()
        self.set_stroke(
            width=min(self.target_stroke_width, max_from_ratio),
            **{"recurse" if config.renderer == "opengl" else "family": False},
        )

        max_from_ratio = self.max_tip_length_to_length_ratio * self.get_arc_length()
        tip_length = min(self.target_tip_length, max_from_ratio)
        for tip in self.tips:
            tip.set_tip_length(tip_length)
            tip.set_stroke_width(self.get_stroke_width())

    def scale(self, factor, scale_tips=False, **kwargs):
        super().scale(factor, **kwargs)
        if not scale_tips:
            self.update_stroke_and_tips()


class Vector(Arrow):
    def __init__(self, direction=RIGHT, buff=0, **kwargs):
        self.buff = buff
        if len(direction) == 2:
            direction = np.hstack([direction, 0])

        super().__init__(ORIGIN, direction, buff=buff, **kwargs)

    def coordinate_label(self, num_decimal_places: int = 0, n_dim: int = 2, **kwargs):
        start = self.get_start()
        end = self.get_end()
        vect = np.round((end - start)[:n_dim], num_decimal_places).reshape((n_dim, 1))
        if num_decimal_places == 0:
            vect = vect.astype(int)
        direction = end.copy()
        direction[1] = 0

        return Matrix(vect, **kwargs).scale(0.8).next_to(end, direction)
