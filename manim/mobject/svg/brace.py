"""Mobject representing curly braces."""

from __future__ import annotations

__all__ = ["Brace", "BraceLabel", "ArcBrace", "BraceText", "BraceBetweenPoints"]

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import svgelements as se

from manim._config import config
from manim.mobject.geometry.arc import Arc
from manim.mobject.geometry.line import Line
from manim.mobject.mobject import Mobject
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.text.tex_mobject import MathTex, Tex

from ...animation.composition import AnimationGroup
from ...animation.fading import FadeIn
from ...animation.growing import GrowFromCenter
from ...constants import *
from ...mobject.types.vectorized_mobject import VMobject
from ...utils.color import BLACK
from ..svg.svg_mobject import VMobjectFromSVGPath

if TYPE_CHECKING:
    from manim.typing import Point3DLike, Vector3D
    from manim.utils.color.core import ParsableManimColor

__all__ = ["Brace", "BraceBetweenPoints", "BraceLabel", "ArcBrace"]


class Brace(VMobjectFromSVGPath):
    """Takes a mobject and draws a brace adjacent to it.

    Passing a direction vector determines the direction from which the
    brace is drawn. By default it is drawn from below.

    Parameters
    ----------
    mobject
        The mobject adjacent to which the brace is placed.
    direction :
        The direction from which the brace faces the mobject.

    See Also
    --------
    :class:`BraceBetweenPoints`

    Examples
    --------
    .. manim:: BraceExample
        :save_last_frame:

        class BraceExample(Scene):
            def construct(self):
                s = Square()
                self.add(s)
                for i in np.linspace(0.1,1.0,4):
                    br = Brace(s, sharpness=i)
                    t = Text(f"sharpness= {i}").next_to(br, RIGHT)
                    self.add(t)
                    self.add(br)
                VGroup(*self.mobjects).arrange(DOWN, buff=0.2)

    """

    def __init__(
        self,
        mobject: Mobject,
        direction: Vector3D | None = DOWN,
        buff: float = 0.2,
        sharpness: float = 2,
        stroke_width: float = 0,
        fill_opacity: float = 1.0,
        background_stroke_width: float = 0,
        background_stroke_color: ParsableManimColor = BLACK,
        **kwargs,
    ):
        path_string_template = (
            "m0.01216 0c-0.01152 0-0.01216 6.103e-4 -0.01216 0.01311v0.007762c0.06776 "
            "0.122 0.1799 0.1455 0.2307 0.1455h{0}c0.03046 3.899e-4 0.07964 0.00449 "
            "0.1246 0.02636 0.0537 0.02695 0.07418 0.05816 0.08648 0.07769 0.001562 "
            "0.002538 0.004539 0.002563 0.01098 0.002563 0.006444-2e-8 0.009421-2.47e-"
            "5 0.01098-0.002563 0.0123-0.01953 0.03278-0.05074 0.08648-0.07769 0.04491"
            "-0.02187 0.09409-0.02597 0.1246-0.02636h{0}c0.05077 0 0.1629-0.02346 "
            "0.2307-0.1455v-0.007762c-1.78e-6 -0.0125-6.365e-4 -0.01311-0.01216-0.0131"
            "1-0.006444-3.919e-8 -0.009348 2.448e-5 -0.01091 0.002563-0.0123 0.01953-"
            "0.03278 0.05074-0.08648 0.07769-0.04491 0.02187-0.09416 0.02597-0.1246 "
            "0.02636h{1}c-0.04786 0-0.1502 0.02094-0.2185 0.1256-0.06833-0.1046-0.1706"
            "-0.1256-0.2185-0.1256h{1}c-0.03046-3.899e-4 -0.07972-0.004491-0.1246-0.02"
            "636-0.0537-0.02695-0.07418-0.05816-0.08648-0.07769-0.001562-0.002538-"
            "0.004467-0.002563-0.01091-0.002563z"
        )
        default_min_width = 0.90552

        self.buff = buff

        angle = -np.arctan2(*direction[:2]) + np.pi
        mobject.rotate(-angle, about_point=ORIGIN)
        left = mobject.get_corner(DOWN + LEFT)
        right = mobject.get_corner(DOWN + RIGHT)
        target_width = right[0] - left[0]
        linear_section_length = max(
            0,
            (target_width * sharpness - default_min_width) / 2,
        )

        path = se.Path(
            path_string_template.format(
                linear_section_length,
                -linear_section_length,
            )
        )

        super().__init__(
            path_obj=path,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            background_stroke_width=background_stroke_width,
            background_stroke_color=background_stroke_color,
            **kwargs,
        )
        self.flip(RIGHT)
        self.stretch_to_fit_width(target_width)
        self.shift(left - self.get_corner(UP + LEFT) + self.buff * DOWN)

        for mob in mobject, self:
            mob.rotate(angle, about_point=ORIGIN)

    def put_at_tip(self, mob: Mobject, use_next_to: bool = True, **kwargs):
        """Puts the given mobject at the brace tip.

        Parameters
        ----------
        mob
            The mobject to be placed at the tip.
        use_next_to
            If true, then :meth:`next_to` is used to place the mobject at the
            tip.
        kwargs
            Any additional keyword arguments are passed to :meth:`next_to` which
            is used to put the mobject next to the brace tip.
        """
        if use_next_to:
            mob.next_to(self.get_tip(), np.round(self.get_direction()), **kwargs)
        else:
            mob.move_to(self.get_tip())
            buff = kwargs.get("buff", DEFAULT_MOBJECT_TO_MOBJECT_BUFFER)
            shift_distance = mob.width / 2.0 + buff
            mob.shift(self.get_direction() * shift_distance)
        return self

    def get_text(self, *text, **kwargs):
        """Places the text at the brace tip.

        Parameters
        ----------
        text
            The text to be placed at the brace tip.
        kwargs
            Any additional keyword arguments are passed to :meth:`.put_at_tip` which
            is used to position the text at the brace tip.

        Returns
        -------
        :class:`~.Tex`
        """
        text_mob = Tex(*text)
        self.put_at_tip(text_mob, **kwargs)
        return text_mob

    def get_tex(self, *tex, **kwargs):
        """Places the tex at the brace tip.

        Parameters
        ----------
        tex
            The tex to be placed at the brace tip.
        kwargs
            Any further keyword arguments are passed to :meth:`.put_at_tip` which
            is used to position the tex at the brace tip.

        Returns
        -------
        :class:`~.MathTex`
        """
        tex_mob = MathTex(*tex)
        self.put_at_tip(tex_mob, **kwargs)
        return tex_mob

    def get_tip(self):
        """Returns the point at the brace tip."""
        # Returns the position of the seventh point in the path, which is the tip.
        if config["renderer"] == "opengl":
            return self.points[34]

        return self.points[28]  # = 7*4

    def get_direction(self):
        """Returns the direction from the center to the brace tip."""
        vect = self.get_tip() - self.get_center()
        return vect / np.linalg.norm(vect)


class BraceLabel(VMobject, metaclass=ConvertToOpenGL):
    """Create a brace with a label attached.

    Parameters
    ----------
    obj
        The mobject adjacent to which the brace is placed.
    text
        The label text.
    brace_direction
        The direction of the brace. By default ``DOWN``.
    label_constructor
        A class or function used to construct a mobject representing
        the label. By default :class:`~.MathTex`.
    font_size
        The font size of the label, passed to the ``label_constructor``.
    buff
        The buffer between the mobject and the brace.
    brace_config
        Arguments to be passed to :class:`.Brace`.
    kwargs
        Additional arguments to be passed to :class:`~.VMobject`.
    """

    def __init__(
        self,
        obj: Mobject,
        text: str,
        brace_direction: np.ndarray = DOWN,
        label_constructor: type = MathTex,
        font_size: float = DEFAULT_FONT_SIZE,
        buff: float = 0.2,
        brace_config: dict | None = None,
        **kwargs,
    ):
        self.label_constructor = label_constructor
        super().__init__(**kwargs)

        self.brace_direction = brace_direction
        if brace_config is None:
            brace_config = {}
        self.brace = Brace(obj, brace_direction, buff, **brace_config)

        if isinstance(text, (tuple, list)):
            self.label = self.label_constructor(*text, font_size=font_size, **kwargs)
        else:
            self.label = self.label_constructor(str(text), font_size=font_size)

        self.brace.put_at_tip(self.label)
        self.add(self.brace, self.label)

    def creation_anim(self, label_anim=FadeIn, brace_anim=GrowFromCenter):
        return AnimationGroup(brace_anim(self.brace), label_anim(self.label))

    def shift_brace(self, obj, **kwargs):
        if isinstance(obj, list):
            obj = self.get_group_class()(*obj)
        self.brace = Brace(obj, self.brace_direction, **kwargs)
        self.brace.put_at_tip(self.label)
        return self

    def change_label(self, *text, **kwargs):
        self.label = self.label_constructor(*text, **kwargs)

        self.brace.put_at_tip(self.label)
        return self

    def change_brace_label(self, obj, *text, **kwargs):
        self.shift_brace(obj)
        self.change_label(*text, **kwargs)
        return self


class BraceText(BraceLabel):
    def __init__(self, obj, text, label_constructor=Tex, **kwargs):
        super().__init__(obj, text, label_constructor=label_constructor, **kwargs)


class BraceBetweenPoints(Brace):
    """Similar to Brace, but instead of taking a mobject it uses 2
    points to place the brace.

    A fitting direction for the brace is
    computed, but it still can be manually overridden.
    If the points go from left to right, the brace is drawn from below.
    Swapping the points places the brace on the opposite side.

    Parameters
    ----------
    point_1 :
        The first point.
    point_2 :
        The second point.
    direction :
        The direction from which the brace faces towards the points.

    Examples
    --------
        .. manim:: BraceBPExample

            class BraceBPExample(Scene):
                def construct(self):
                    p1 = [0,0,0]
                    p2 = [1,2,0]
                    brace = BraceBetweenPoints(p1,p2)
                    self.play(Create(NumberPlane()))
                    self.play(Create(brace))
                    self.wait(2)
    """

    def __init__(
        self,
        point_1: Point3DLike | None,
        point_2: Point3DLike | None,
        direction: Vector3D | None = ORIGIN,
        **kwargs,
    ):
        if all(direction == ORIGIN):
            line_vector = np.array(point_2) - np.array(point_1)
            direction = np.array([line_vector[1], -line_vector[0], 0])
        super().__init__(Line(point_1, point_2), direction=direction, **kwargs)


class ArcBrace(Brace):
    """Creates a :class:`~Brace` that wraps around an :class:`~.Arc`.

    The direction parameter allows the brace to be applied
    from outside or inside the arc.

    .. warning::
        The :class:`ArcBrace` is smaller for arcs with smaller radii.

    .. note::
        The :class:`ArcBrace` is initially a vertical :class:`Brace` defined by the
        length of the :class:`~.Arc`, but is scaled down to match the start and end
        angles. An exponential function is then applied after it is shifted based on
        the radius of the arc.

        The scaling effect is not applied for arcs with radii smaller than 1 to prevent
        over-scaling.

    Parameters
    ----------
    arc
        The :class:`~.Arc` that wraps around the :class:`Brace` mobject.
    direction
        The direction from which the brace faces the arc.
        ``LEFT`` for inside the arc, and ``RIGHT`` for the outside.

    Example
    -------
        .. manim:: ArcBraceExample
            :save_last_frame:
            :ref_classes: Arc

            class ArcBraceExample(Scene):
                def construct(self):
                    arc_1 = Arc(radius=1.5,start_angle=0,angle=2*PI/3).set_color(RED)
                    brace_1 = ArcBrace(arc_1,LEFT)
                    group_1 = VGroup(arc_1,brace_1)

                    arc_2 = Arc(radius=3,start_angle=0,angle=5*PI/6).set_color(YELLOW)
                    brace_2 = ArcBrace(arc_2)
                    group_2 = VGroup(arc_2,brace_2)

                    arc_3 = Arc(radius=0.5,start_angle=-0,angle=PI).set_color(BLUE)
                    brace_3 = ArcBrace(arc_3)
                    group_3 = VGroup(arc_3,brace_3)

                    arc_4 = Arc(radius=0.2,start_angle=0,angle=3*PI/2).set_color(GREEN)
                    brace_4 = ArcBrace(arc_4)
                    group_4 = VGroup(arc_4,brace_4)

                    arc_group = VGroup(group_1, group_2, group_3, group_4).arrange_in_grid(buff=1.5)
                    self.add(arc_group.center())

    """

    def __init__(
        self,
        arc: Arc | None = None,
        direction: Sequence[float] = RIGHT,
        **kwargs,
    ):
        if arc is None:
            arc = Arc(start_angle=-1, angle=2, radius=1)
        arc_end_angle = arc.start_angle + arc.angle
        line = Line(UP * arc.start_angle, UP * arc_end_angle)
        scale_shift = RIGHT * np.log(arc.radius)

        if arc.radius >= 1:
            line.scale(arc.radius, about_point=ORIGIN)
            super().__init__(line, direction=direction, **kwargs)
            self.scale(1 / (arc.radius), about_point=ORIGIN)
        else:
            super().__init__(line, direction=direction, **kwargs)

        if arc.radius >= 0.3:
            self.shift(scale_shift)
        else:
            self.shift(RIGHT * np.log(0.3))

        self.apply_complex_function(np.exp)
        self.shift(arc.get_arc_center())
