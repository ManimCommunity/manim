__all__ = [
    "TipableVMobject",
    "CurvedArrow",
    "CurvedDoubleArrow",
    "Arrow",
    "Vector",
    "DoubleArrow",
    "ArrowTip",
]

from typing import Literal, Optional, Union
from functools import wraps

import numpy as np
from colour import Color

from ..constants import *
from ..utils.color import WHITE
from ..utils.space_ops import angle_of_vector, normalize

from .geometry import Triangle
from .mobject import Mobject
from .types.vectorized_mobject import MetaVMobject, VMobject
from .opengl_mobject import OpenGLMobject
from .types.opengl_vectorized_mobject import OpenGLVMobject

DEFAULT_ARROW_TO_STROKE_WIDTH_RATIO = 5.833333333


class ArrowTip:  # TODO: add presets via string
    def __init__(
        self,
        base_line: MetaVMobject,
        mobject: Optional[Union[Mobject, OpenGLMobject]] = None,
        *,
        relative_position: float = 1,
        tip_angle: float = PI / 2,
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
        self.relative_position = relative_position
        self.secant_delta = secant_delta

        self.tip_angle = tip_angle
        self.tip_alignment = tip_alignment

        # ignore scale_auto if length and width are defined
        if length is not None and width is not None:
            self.set_length(length, update=False)
            self.set_width(width, update=False)

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
            self.set_width(width, update=False, proportional=True)

        if color:
            if color == "copy":
                color = base_line.get_stroke_color()
            mobject.set_color(color)
        if filled is not None and isinstance(mobject, (VMobject, OpenGLVMobject)):
            mobject.set_fill(opacity=float(filled))

        self.update_positioning()
        base_line.add(mobject)
        base_line.tips.append(self)

    def _tip_shape_editing(func):
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

    @_tip_shape_editing
    def set_length(self, length, proportional=False):
        if proportional:
            self.mobject.width = length
        else:
            self.mobject.stretch_to_fit_width(length)

    @_tip_shape_editing
    def set_width(self, width, proportional=False):
        if proportional:
            self.mobject.height = width
        else:
            self.mobject.stretch_to_fit_height(width)

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


class Arrow:  # Line
    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        *,
        buff=MED_SMALL_BUFF,
        path_arc=None,
        stroke_width=6,
        max_tip_length_to_length_ratio=0.25,
        max_stroke_width_to_length_ratio=5,
        preserve_tip_size_when_scaling=True,
    ):
        pass


# class Arrow(Line):
#     """An arrow.

#     Parameters
#     ----------
#     args : Any
#         Arguments to be passed to :class:`Line`.
#     stroke_width : :class:`float`, optional
#         The thickness of the arrow. Influenced by :attr:`max_stroke_width_to_length_ratio`.
#     buff : :class:`float`, optional
#         The distance of the arrow from its start and end points.
#     max_tip_length_to_length_ratio : :class:`float`, optional
#         :attr:`tip_length` scales with the length of the arrow. Increasing this ratio raises the max value of :attr:`tip_length`.
#     max_stroke_width_to_length_ratio : :class:`float`, optional
#         :attr:`stroke_width` scales with the length of the arrow. Increasing this ratio ratios the max value of :attr:`stroke_width`.
#     preserve_tip_size_when_scaling : :class:`bool`, optional
#         No purpose.
#     kwargs : Any
#         Additional arguments to be passed to :class:`Line`.

#     Examples
#     --------

#     .. manim:: ArrowExample
#         :save_last_frame:

#         from manim.mobject.geometry import ArrowSquareTip
#         class ArrowExample(Scene):
#             def construct(self):
#                 arrow_1 = Arrow(start=RIGHT, end=LEFT, color=GOLD)
#                 arrow_2 = Arrow(start=RIGHT, end=LEFT, color=GOLD, tip_shape=ArrowSquareTip).shift(DOWN)
#                 g1 = Group(arrow_1, arrow_2)

#                 # the effect of buff
#                 square = Square(color=MAROON_A)
#                 arrow_3 = Arrow(start=LEFT, end=RIGHT)
#                 arrow_4 = Arrow(start=LEFT, end=RIGHT, buff=0).next_to(arrow_1, UP)
#                 g2 = Group(arrow_3, arrow_4, square)

#                 # a shorter arrow has a shorter tip and smaller stroke width
#                 arrow_5 = Arrow(start=ORIGIN, end=config.top).shift(LEFT * 4)
#                 arrow_6 = Arrow(start=config.top + DOWN, end=config.top).shift(LEFT * 3)
#                 g3 = Group(arrow_5, arrow_6)

#                 self.add(Group(g1, g2, g3).arrange(buff=2))

#     See Also
#     --------
#     :class:`ArrowTip`
#     :class:`CurvedArrow`
#     """

#     def __init__(
#         self,
#         *args,
#         stroke_width=6,
#         buff=MED_SMALL_BUFF,
#         max_tip_length_to_length_ratio=0.25,
#         max_stroke_width_to_length_ratio=5,
#         preserve_tip_size_when_scaling=True,
#         **kwargs,
#     ):
#         self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
#         self.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio
#         self.preserve_tip_size_when_scaling = (
#             preserve_tip_size_when_scaling  # is this used anywhere
#         )
#         tip_shape = kwargs.pop("tip_shape", ArrowTriangleFilledTip)
#         super().__init__(*args, buff=buff, stroke_width=stroke_width, **kwargs)
#         # TODO, should this be affected when
#         # Arrow.set_stroke is called?
#         self.initial_stroke_width = self.stroke_width
#         self.add_tip(tip_shape=tip_shape)
#         self.set_stroke_width_from_length()

#     def scale(self, factor, scale_tips=False, **kwargs):
#         r"""Scale an arrow, but keep stroke width and arrow tip size fixed.

#         See Also
#         --------
#         :meth:`~.Mobject.scale`

#         Examples
#         --------
#         ::

#             >>> arrow = Arrow(np.array([-1, -1, 0]), np.array([1, 1, 0]), buff=0)
#             >>> scaled_arrow = arrow.scale(2)
#             >>> np.round(scaled_arrow.get_start_and_end(), 8) + 0
#             array([[-2., -2.,  0.],
#                    [ 2.,  2.,  0.]])
#             >>> arrow.tip.length == scaled_arrow.tip.length
#             True

#         Manually scaling the object using the default method
#         :meth:`~.Mobject.scale` does not have the same properties::

#             >>> new_arrow = Arrow(np.array([-1, -1, 0]), np.array([1, 1, 0]), buff=0)
#             >>> another_scaled_arrow = VMobject.scale(new_arrow, 2)
#             >>> another_scaled_arrow.tip.length == arrow.tip.length
#             False

#         """
#         if self.get_length() == 0:
#             return self

#         if scale_tips:
#             super().scale(factor, **kwargs)
#             self.set_stroke_width_from_length()
#             return self

#         has_tip = self.has_tip()
#         has_start_tip = self.has_start_tip()
#         if has_tip or has_start_tip:
#             old_tips = self.pop_tips()

#         super().scale(factor, **kwargs)
#         self.set_stroke_width_from_length()

#         if has_tip:
#             self.add_tip(tip=old_tips[0])
#         if has_start_tip:
#             self.add_tip(tip=old_tips[1], at_start=True)
#         return self

#     def get_normal_vector(self) -> np.ndarray:
#         """Returns the normal of a vector.

#         Examples
#         --------
#         ::

#             >>> Arrow().get_normal_vector() + 0. # add 0. to avoid negative 0 in output
#             array([ 0.,  0., -1.])
#         """

#         p0, p1, p2 = self.tip.get_start_anchors()[:3]
#         return normalize(np.cross(p2 - p1, p1 - p0))

#     def reset_normal_vector(self):
#         """Resets the normal of a vector"""
#         self.normal_vector = self.get_normal_vector()
#         return self

#     def get_default_tip_length(self) -> float:
#         """Returns the default tip_length of the arrow.

#         Examples
#         --------

#         ::

#             >>> Arrow().get_default_tip_length()
#             0.35
#         """

#         max_ratio = self.max_tip_length_to_length_ratio
#         return min(self.tip_length, max_ratio * self.get_length())

#     def set_stroke_width_from_length(self):
#         """Used internally. Sets stroke width based on length."""
#         max_ratio = self.max_stroke_width_to_length_ratio
#         if config.renderer == "opengl":
#             self.set_stroke(
#                 width=min(self.initial_stroke_width, max_ratio * self.get_length()),
#                 recurse=False,
#             )
#         else:
#             self.set_stroke(
#                 width=min(self.initial_stroke_width, max_ratio * self.get_length()),
#                 family=False,
#             )
#         return self


# class CurvedArrow(ArcBetweenPoints):
#     def __init__(self, start_point, end_point, **kwargs):
#         super().__init__(start_point, end_point, **kwargs)
#         self.add_tip(tip_shape=kwargs.pop("tip_shape", ArrowTriangleFilledTip))


# class CurvedDoubleArrow(CurvedArrow):
#     def __init__(self, start_point, end_point, **kwargs):
#         if "tip_shape_end" in kwargs:
#             kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
#         tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
#         super().__init__(start_point, end_point, **kwargs)
#         self.add_tip(at_start=True, tip_shape=tip_shape_start)


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

    def coordinate_label(
        self, integer_labels: bool = True, n_dim: int = 2, color: str = WHITE
    ):
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

        vect = np.array(self.get_end())
        if integer_labels:
            vect = np.round(vect).astype(int)
        vect = vect[:n_dim]
        vect = vect.reshape((n_dim, 1))

        label = Matrix(vect)
        label.scale(LARGE_BUFF - 0.2)

        shift_dir = np.array(self.get_end())
        if shift_dir[0] >= 0:  # Pointing right
            shift_dir -= label.get_left() + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * LEFT
        else:  # Pointing left
            shift_dir -= label.get_right() + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * RIGHT
        label.shift(shift_dir)
        label.set_color(color)
        return label


# class DoubleArrow(Arrow):
#     """An arrow with tips on both ends.

#     Parameters
#     ----------
#     args : Any
#         Arguments to be passed to :class:`Arrow`
#     kwargs : Any
#         Additional arguments to be passed to :class:`Arrow`

#     Examples
#     --------

#     .. manim:: DoubleArrowExample
#         :save_last_frame:

#         from manim.mobject.geometry import ArrowCircleFilledTip
#         class DoubleArrowExample(Scene):
#             def construct(self):
#                 circle = Circle(radius=2.0)
#                 d_arrow = DoubleArrow(start=circle.get_left(), end=circle.get_right())
#                 d_arrow_2 = DoubleArrow(tip_shape_end=ArrowCircleFilledTip, tip_shape_start=ArrowCircleFilledTip)
#                 group = Group(Group(circle, d_arrow), d_arrow_2).arrange(UP, buff=1)
#                 self.add(group)

#     See Also
#     --------
#     :class:`ArrowTip`
#     :class:`CurvedDoubleArrow`
#     """

#     def __init__(self, *args, **kwargs):
#         if "tip_shape_end" in kwargs:
#             kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
#         tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
#         super().__init__(*args, **kwargs)
#         self.add_tip(at_start=True, tip_shape=tip_shape_start)


# class ArrowTip(metaclass=MetaVMobject):
#     r"""Base class for arrow tips.

#     See Also
#     --------
#     :class:`ArrowTriangleTip`
#     :class:`ArrowTriangleFilledTip`
#     :class:`ArrowCircleTip`
#     :class:`ArrowCircleFilledTip`
#     :class:`ArrowSquareTip`
#     :class:`ArrowSquareFilledTip`

#     Examples
#     --------
#     Cannot be used directly, only intended for inheritance::

#         >>> tip = ArrowTip()
#         Traceback (most recent call last):
#         ...
#         NotImplementedError: Has to be implemented in inheriting subclasses.

#     Instead, use one of the pre-defined ones, or make
#     a custom one like this:

#     .. manim:: CustomTipExample

#         >>> class MyCustomArrowTip(ArrowTip, RegularPolygon):
#         ...     def __init__(self, **kwargs):
#         ...         RegularPolygon.__init__(self, n=5, **kwargs)
#         ...         length = 0.35
#         ...         self.width = length
#         ...         self.stretch_to_fit_height(length)
#         >>> arr = Arrow(np.array([-2, -2, 0]), np.array([2, 2, 0]),
#         ...             tip_shape=MyCustomArrowTip)
#         >>> isinstance(arr.tip, RegularPolygon)
#         True
#         >>> from manim import Scene
#         >>> class CustomTipExample(Scene):
#         ...     def construct(self):
#         ...         self.play(Create(arr))

#     Using a class inherited from :class:`ArrowTip` to get a non-filled
#     tip is a shorthand to manually specifying the arrow tip style as follows::

#         >>> arrow = Arrow(np.array([0, 0, 0]), np.array([1, 1, 0]),
#         ...               tip_style={'fill_opacity': 0, 'stroke_width': 3})

#     The following example illustrates the usage of all of the predefined
#     arrow tips.

#     .. manim:: ArrowTipsShowcase
#         :save_last_frame:

#         from manim.mobject.geometry import ArrowTriangleTip, ArrowSquareTip, ArrowSquareFilledTip,\
#                                         ArrowCircleTip, ArrowCircleFilledTip
#         class ArrowTipsShowcase(Scene):
#             def construct(self):
#                 a00 = Arrow(start=[-2, 3, 0], end=[2, 3, 0], color=YELLOW)
#                 a11 = Arrow(start=[-2, 2, 0], end=[2, 2, 0], tip_shape=ArrowTriangleTip)
#                 a12 = Arrow(start=[-2, 1, 0], end=[2, 1, 0])
#                 a21 = Arrow(start=[-2, 0, 0], end=[2, 0, 0], tip_shape=ArrowSquareTip)
#                 a22 = Arrow([-2, -1, 0], [2, -1, 0], tip_shape=ArrowSquareFilledTip)
#                 a31 = Arrow([-2, -2, 0], [2, -2, 0], tip_shape=ArrowCircleTip)
#                 a32 = Arrow([-2, -3, 0], [2, -3, 0], tip_shape=ArrowCircleFilledTip)
#                 b11 = a11.copy().scale(0.5, scale_tips=True).next_to(a11, RIGHT)
#                 b12 = a12.copy().scale(0.5, scale_tips=True).next_to(a12, RIGHT)
#                 b21 = a21.copy().scale(0.5, scale_tips=True).next_to(a21, RIGHT)
#                 self.add(a00, a11, a12, a21, a22, a31, a32, b11, b12, b21)

#     """

#     def __init__(self, *args, **kwargs):
#         raise NotImplementedError("Has to be implemented in inheriting subclasses.")

#     @property
#     def base(self):
#         r"""The base point of the arrow tip.

#         This is the point connecting to the arrow line.

#         Examples
#         --------
#         ::

#             >>> arrow = Arrow(np.array([0, 0, 0]), np.array([2, 0, 0]), buff=0)
#             >>> arrow.tip.base.round(2) + 0.  # add 0. to avoid negative 0 in output
#             array([1.65, 0.  , 0.  ])

#         """
#         return self.point_from_proportion(0.5)

#     @property
#     def tip_point(self):
#         r"""The tip point of the arrow tip.

#         Examples
#         --------
#         ::

#             >>> arrow = Arrow(np.array([0, 0, 0]), np.array([2, 0, 0]), buff=0)
#             >>> arrow.tip.tip_point.round(2) + 0.
#             array([2., 0., 0.])

#         """
#         return self.get_points()[0]

#     @property
#     def vector(self):
#         r"""The vector pointing from the base point to the tip point.

#         Examples
#         --------
#         ::

#             >>> arrow = Arrow(np.array([0, 0, 0]), np.array([2, 2, 0]), buff=0)
#             >>> arrow.tip.vector.round(2) + 0.
#             array([0.25, 0.25, 0.  ])

#         """
#         return self.tip_point - self.base

#     @property
#     def tip_angle(self):
#         r"""The angle of the arrow tip.

#         Examples
#         --------
#         ::

#             >>> arrow = Arrow(np.array([0, 0, 0]), np.array([1, 1, 0]), buff=0)
#             >>> round(arrow.tip.tip_angle, 5) == round(PI/4, 5)
#             True

#         """
#         return angle_of_vector(self.vector)

#     @property
#     def length(self):
#         r"""The length of the arrow tip.

#         Examples
#         --------
#         ::

#             >>> arrow = Arrow(np.array([0, 0, 0]), np.array([1, 2, 0]))
#             >>> round(arrow.tip.length, 3)
#             0.35

#         """
#         return np.linalg.norm(self.vector)


# class ArrowTriangleTip(ArrowTip, Triangle):
#     r"""Triangular arrow tip."""

#     def __init__(
#         self,
#         fill_opacity=0,
#         stroke_width=3,
#         length=DEFAULT_ARROW_TIP_LENGTH,
#         start_angle=PI,
#         **kwargs,
#     ):
#         Triangle.__init__(
#             self,
#             fill_opacity=fill_opacity,
#             stroke_width=stroke_width,
#             start_angle=start_angle,
#             **kwargs,
#         )
#         self.width = length
#         self.stretch_to_fit_height(length)


# class ArrowTriangleFilledTip(ArrowTriangleTip):
#     r"""Triangular arrow tip with filled tip.

#     This is the default arrow tip shape.
#     """

#     def __init__(self, fill_opacity=1, stroke_width=0, **kwargs):
#         super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)


# class ArrowCircleTip(ArrowTip, Circle):
#     r"""Circular arrow tip."""

#     def __init__(
#         self,
#         fill_opacity=0,
#         stroke_width=3,
#         length=DEFAULT_ARROW_TIP_LENGTH,
#         start_angle=PI,
#         **kwargs,
#     ):
#         self.start_angle = start_angle
#         Circle.__init__(
#             self, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
#         )
#         self.width = length
#         self.stretch_to_fit_height(length)


# class ArrowCircleFilledTip(ArrowCircleTip):
#     r"""Circular arrow tip with filled tip."""

#     def __init__(self, fill_opacity=1, stroke_width=0, **kwargs):
#         super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)


# class ArrowSquareTip(ArrowTip, Square):
#     r"""Square arrow tip."""

#     def __init__(
#         self,
#         fill_opacity=0,
#         stroke_width=3,
#         length=DEFAULT_ARROW_TIP_LENGTH,
#         start_angle=PI,
#         **kwargs,
#     ):
#         self.start_angle = start_angle
#         Square.__init__(
#             self,
#             fill_opacity=fill_opacity,
#             stroke_width=stroke_width,
#             side_length=length,
#             **kwargs,
#         )
#         self.width = length
#         self.stretch_to_fit_height(length)


# class ArrowSquareFilledTip(ArrowSquareTip):
#     r"""Square arrow tip with filled tip."""

#     def __init__(self, fill_opacity=1, stroke_width=0, **kwargs):
#         super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)
