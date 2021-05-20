__all__ = [
    "Arrow",
]

from functools import wraps
from typing import Literal, Optional, Union

import numpy as np
from colour import Color

from .. import config
from ..constants import *
from ..utils.color import WHITE
from ..utils.space_ops import angle_of_vector, normalize
from .geometry import Line, Triangle
from .mobject import Mobject
from .opengl_mobject import OpenGLMobject
from .types.opengl_vectorized_mobject import OpenGLVMobject
from .types.vectorized_mobject import MetaVMobject, VMobject

DEFAULT_ARROW_TO_STROKE_WIDTH_RATIO = 35 / 6
# TODO needs cleanup


class ArrowTip:  # TODO: add presets via string
    def __init__(
        self,
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
        secant_delta: float = 1e-4,
        **kwargs,
    ):
        if mobject is None:
            mobject = Triangle()
            mobject.width = DEFAULT_ARROW_TIP_LENGTH
            mobject.stretch_to_fit_height(DEFAULT_ARROW_TIP_LENGTH)
            filled = filled is None or filled
            color = "copy" if color is None else color
        super().__init__(**kwargs)
        self.base_line = base_line
        self.mobject = mobject
        if backwards:
            relative_position = 1 - relative_position
            tip_angle += PI
            tip_alignment = tip_alignment * -1

        self.relative_position = relative_position
        self.secant_delta = secant_delta

        self.tip_angle = tip_angle
        self.tip_alignment = tip_alignment

        # ignore scale_auto if length and width are defined
        if length is not None and width is not None:
            self.set_length(length, proportional=False, update=False)
            self.set_width(width, proportional=False, update=False)

        # use scale_auto to decide if scaling is proportional if only one dim is defined.
        elif length is not None:
            self.set_length(length, proportional=scale_auto, update=False)
        elif width is not None:
            self.set_width(length, proportional=scale_auto, update=False)

        # choose width depending on base line stroke width
        elif scale_auto:
            width = (
                self.base_line.stroke_width * DEFAULT_ARROW_TO_STROKE_WIDTH_RATIO / 100
            )
            self.set_width(width, update=False)

        if color:
            if color == "copy":
                color = base_line.get_stroke_color()
            mobject.set_color(color)
        if filled is not None and isinstance(mobject, (VMobject, OpenGLVMobject)):
            mobject.set_fill(opacity=float(filled))

        self.update_positioning()
        base_line.add(mobject)
        base_line.tips.append(self)

    def _unrotated_tip(func):
        @wraps(func)
        def func_wrapper(self, *args, update=True, **kwargs):
            if self.tip_angle != 0:
                self.mobject.rotate(-self.tip_angle)
                self.tip_angle = 0
            result = func(self, *args, **kwargs)
            if update:
                self.update_positioning()
            return result

        return func_wrapper

    @_unrotated_tip
    def set_length(self, length, proportional=True):
        if proportional:
            self.mobject.width = length
        else:
            self.mobject.stretch_to_fit_width(length)

    @_unrotated_tip
    def get_length(self):
        return self.mobject.width

    @_unrotated_tip
    def set_width(self, width, proportional=True):
        if proportional:
            self.mobject.height = width
        else:
            self.mobject.stretch_to_fit_height(width)

    @_unrotated_tip
    def get_width(self):
        return self.mobject.height

    def update_positioning(self, scene=None):
        p = [
            self.base_line.point_from_proportion(
                np.clip(self.relative_position + d, 0, 1)
            )
            for d in [-self.secant_delta, 0, self.secant_delta]
        ]
        angle = angle_of_vector(p[2] - p[0])

        self.mobject.rotate(-self.tip_angle)
        self.mobject.move_to(p[1], self.tip_alignment)
        rotation_about_point = self.mobject.get_critical_point(self.tip_alignment)
        self.mobject.rotate(angle, about_point=rotation_about_point)
        self.tip_angle = angle


class Arrow(Line):  # Line
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
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio
        self.target_stroke_width = target_stroke_width

        kwargs.setdefault("tip_alignment", RIGHT)
        self.add_tip(None if tip_mobject is None else tip_mobject.copy(), **kwargs)
        if double:
            self.add_tip(
                None if tip_mobject is None else tip_mobject.copy(),
                backwards=True,
                **kwargs,
            )
        self.target_tip_length = self.tips[0].get_length()

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
            tip.set_length(tip_length)
            tip.mobject.set_stroke_width(self.get_stroke_width())


class Vector(Arrow):
    """A vector specialized for use in graphs.

    Parameters
    ----------
    direction : Union[:class:`list`, :class:`numpy.ndarray`]
        The direction of the arrow.
    buff : :class:`float`
         The distance of the vector from its endpoints.
    kwargs : Any
        Additional arguments to be passed to :class:`Arrow`

    Examples
    --------

    .. manim:: VectorExample
        :save_last_frame:

        class VectorExample(Scene):
            def construct(self):
                plane = NumberPlane()
                vector_1 = Vector([1,2])
                vector_2 = Vector([-5,-2])
                self.add(plane, vector_1, vector_2)
    """

    def __init__(self, direction=RIGHT, buff=0, **kwargs):
        self.buff = buff
        if len(direction) == 2:
            direction = np.hstack([direction, 0])

        super().__init__(ORIGIN, direction, buff=buff, **kwargs)

    def coordinate_label(self, num_decimal_places: int = 0, n_dim: int = 2, **kwargs):
        """Creates a label based on the coordinates of the vector.

        Parameters
        ----------
        integer_labels
            Whether or not to round the coordinates to integers.
        n_dim
            The number of dimensions of the vector.
        color
            The color of the label.

        Examples
        --------

        .. manim VectorCoordinateLabel
            :save_last_frame:

            class VectorCoordinateLabel(Scene):
                def construct(self):
                    plane = NumberPlane()

                    vect_1 = Vector([1, 2])
                    vect_2 = Vector([-3, -2])
                    label_1 = vect1.coordinate_label()
                    label_2 = vect2.coordinate_label(color=YELLOW)

                    self.add(plane, vect_1, vect_2, label_1, label_2)
        """
        # avoiding circular imports
        from .matrix import Matrix

        end = self.point_from_proportion(1)
        vect = np.round(end[:n_dim], num_decimal_places).reshape((n_dim, 1))
        if num_decimal_places == 0:
            vect = vect.astype(int)
        direction = end.copy()
        direction[1] = 0

        return Matrix(vect, **kwargs).scale(0.8).next_to(end, direction)
