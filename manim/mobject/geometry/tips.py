r"""A collection of tip mobjects for use with :class:`~.TipableVMobject`."""

from __future__ import annotations

__all__ = [
    "ArrowTip",
    "ArrowCircleFilledTip",
    "ArrowCircleTip",
    "ArrowSquareTip",
    "ArrowSquareFilledTip",
]

import numpy as np

from manim.constants import *
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square, Triangle
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.space_ops import angle_of_vector


class ArrowTip(VMobject, metaclass=ConvertToOpenGL):
    r"""Base class for arrow tips.

    .. seealso::
        :class:`ArrowTriangleTip`
        :class:`ArrowTriangleFilledTip`
        :class:`ArrowCircleTip`
        :class:`ArrowCircleFilledTip`
        :class:`ArrowSquareTip`
        :class:`ArrowSquareFilledTip`

    Examples
    --------
    Cannot be used directly, only intended for inheritance::

        >>> tip = ArrowTip()
        Traceback (most recent call last):
        ...
        NotImplementedError: Has to be implemented in inheriting subclasses.

    Instead, use one of the pre-defined ones, or make
    a custom one like this:

    .. manim:: CustomTipExample

        >>> from manim import RegularPolygon, Arrow
        >>> class MyCustomArrowTip(ArrowTip, RegularPolygon):
        ...     def __init__(self, length=0.35, **kwargs):
        ...         RegularPolygon.__init__(self, n=5, **kwargs)
        ...         self.width = length
        ...         self.stretch_to_fit_height(length)
        >>> arr = Arrow(np.array([-2, -2, 0]), np.array([2, 2, 0]),
        ...             tip_shape=MyCustomArrowTip)
        >>> isinstance(arr.tip, RegularPolygon)
        True
        >>> from manim import Scene, Create
        >>> class CustomTipExample(Scene):
        ...     def construct(self):
        ...         self.play(Create(arr))

    Using a class inherited from :class:`ArrowTip` to get a non-filled
    tip is a shorthand to manually specifying the arrow tip style as follows::

        >>> arrow = Arrow(np.array([0, 0, 0]), np.array([1, 1, 0]),
        ...               tip_style={'fill_opacity': 0, 'stroke_width': 3})

    The following example illustrates the usage of all of the predefined
    arrow tips.

    .. manim:: ArrowTipsShowcase
        :save_last_frame:

        from manim.mobject.geometry.tips import ArrowTriangleTip,\
                                                ArrowSquareTip, ArrowSquareFilledTip,\
                                                ArrowCircleTip, ArrowCircleFilledTip
        class ArrowTipsShowcase(Scene):
            def construct(self):
                a00 = Arrow(start=[-2, 3, 0], end=[2, 3, 0], color=YELLOW)
                a11 = Arrow(start=[-2, 2, 0], end=[2, 2, 0], tip_shape=ArrowTriangleTip)
                a12 = Arrow(start=[-2, 1, 0], end=[2, 1, 0])
                a21 = Arrow(start=[-2, 0, 0], end=[2, 0, 0], tip_shape=ArrowSquareTip)
                a22 = Arrow([-2, -1, 0], [2, -1, 0], tip_shape=ArrowSquareFilledTip)
                a31 = Arrow([-2, -2, 0], [2, -2, 0], tip_shape=ArrowCircleTip)
                a32 = Arrow([-2, -3, 0], [2, -3, 0], tip_shape=ArrowCircleFilledTip)
                b11 = a11.copy().scale(0.5, scale_tips=True).next_to(a11, RIGHT)
                b12 = a12.copy().scale(0.5, scale_tips=True).next_to(a12, RIGHT)
                b21 = a21.copy().scale(0.5, scale_tips=True).next_to(a21, RIGHT)
                self.add(a00, a11, a12, a21, a22, a31, a32, b11, b12, b21)

    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Has to be implemented in inheriting subclasses.")

    @property
    def base(self):
        r"""The base point of the arrow tip.

        This is the point connecting to the arrow line.

        Examples
        --------
        ::

            >>> from manim import Arrow
            >>> arrow = Arrow(np.array([0, 0, 0]), np.array([2, 0, 0]), buff=0)
            >>> arrow.tip.base.round(2) + 0.  # add 0. to avoid negative 0 in output
            array([1.65, 0.  , 0.  ])

        """
        return self.point_from_proportion(0.5)

    @property
    def tip_point(self):
        r"""The tip point of the arrow tip.

        Examples
        --------
        ::

            >>> from manim import Arrow
            >>> arrow = Arrow(np.array([0, 0, 0]), np.array([2, 0, 0]), buff=0)
            >>> arrow.tip.tip_point.round(2) + 0.
            array([2., 0., 0.])

        """
        return self.points[0]

    @property
    def vector(self):
        r"""The vector pointing from the base point to the tip point.

        Examples
        --------
        ::

            >>> from manim import Arrow
            >>> arrow = Arrow(np.array([0, 0, 0]), np.array([2, 2, 0]), buff=0)
            >>> arrow.tip.vector.round(2) + 0.
            array([0.25, 0.25, 0.  ])

        """
        return self.tip_point - self.base

    @property
    def tip_angle(self):
        r"""The angle of the arrow tip.

        Examples
        --------
        ::

            >>> from manim import Arrow
            >>> arrow = Arrow(np.array([0, 0, 0]), np.array([1, 1, 0]), buff=0)
            >>> round(arrow.tip.tip_angle, 5) == round(PI/4, 5)
            True

        """
        return angle_of_vector(self.vector)

    @property
    def length(self):
        r"""The length of the arrow tip.

        Examples
        --------
        ::

            >>> from manim import Arrow
            >>> arrow = Arrow(np.array([0, 0, 0]), np.array([1, 2, 0]))
            >>> round(arrow.tip.length, 3)
            0.35

        """
        return np.linalg.norm(self.vector)


class ArrowTriangleTip(ArrowTip, Triangle):
    r"""Triangular arrow tip."""

    def __init__(
        self,
        fill_opacity=0,
        stroke_width=3,
        length=DEFAULT_ARROW_TIP_LENGTH,
        width=DEFAULT_ARROW_TIP_LENGTH,
        start_angle=PI,
        **kwargs,
    ):
        Triangle.__init__(
            self,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            start_angle=start_angle,
            **kwargs,
        )
        self.width = width

        self.stretch_to_fit_width(length)
        self.stretch_to_fit_height(width)


class ArrowTriangleFilledTip(ArrowTriangleTip):
    r"""Triangular arrow tip with filled tip.

    This is the default arrow tip shape.
    """

    def __init__(self, fill_opacity=1, stroke_width=0, **kwargs):
        super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)


class ArrowCircleTip(ArrowTip, Circle):
    r"""Circular arrow tip."""

    def __init__(
        self,
        fill_opacity=0,
        stroke_width=3,
        length=DEFAULT_ARROW_TIP_LENGTH,
        start_angle=PI,
        **kwargs,
    ):
        self.start_angle = start_angle
        Circle.__init__(
            self, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )
        self.width = length
        self.stretch_to_fit_height(length)


class ArrowCircleFilledTip(ArrowCircleTip):
    r"""Circular arrow tip with filled tip."""

    def __init__(self, fill_opacity=1, stroke_width=0, **kwargs):
        super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)


class ArrowSquareTip(ArrowTip, Square):
    r"""Square arrow tip."""

    def __init__(
        self,
        fill_opacity=0,
        stroke_width=3,
        length=DEFAULT_ARROW_TIP_LENGTH,
        start_angle=PI,
        **kwargs,
    ):
        self.start_angle = start_angle
        Square.__init__(
            self,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            side_length=length,
            **kwargs,
        )
        self.width = length
        self.stretch_to_fit_height(length)


class ArrowSquareFilledTip(ArrowSquareTip):
    r"""Square arrow tip with filled tip."""

    def __init__(self, fill_opacity=1, stroke_width=0, **kwargs):
        super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)
