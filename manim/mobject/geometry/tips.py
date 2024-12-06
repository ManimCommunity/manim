r"""A collection of tip mobjects for use with :class:`~.TipableVMobject`."""

from __future__ import annotations

__all__ = [
    "ArrowTip",
    "ArrowCircleFilledTip",
    "ArrowCircleTip",
    "ArrowSquareTip",
    "ArrowSquareFilledTip",
    "ArrowTriangleTip",
    "ArrowTriangleFilledTip",
    "StealthTip",
]

from typing import TYPE_CHECKING

import numpy as np

from manim.constants import *
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square, Triangle
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VMobject
from manim.utils.space_ops import angle_of_vector

if TYPE_CHECKING:
    from typing import Any

    from manim.typing import Point3D, Vector3D


class ArrowTip(VMobject, metaclass=ConvertToOpenGL):
    r"""Base class for arrow tips.

    .. seealso::
        :class:`ArrowTriangleTip`
        :class:`ArrowTriangleFilledTip`
        :class:`ArrowCircleTip`
        :class:`ArrowCircleFilledTip`
        :class:`ArrowSquareTip`
        :class:`ArrowSquareFilledTip`
        :class:`StealthTip`

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
        >>> arr = Arrow(
        ...     np.array([-2, -2, 0]), np.array([2, 2, 0]), tip_shape=MyCustomArrowTip
        ... )
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

        class ArrowTipsShowcase(Scene):
            def construct(self):
                tip_names = [
                    'Default (YELLOW)', 'ArrowTriangleTip', 'Default', 'ArrowSquareTip',
                    'ArrowSquareFilledTip', 'ArrowCircleTip', 'ArrowCircleFilledTip', 'StealthTip'
                ]

                big_arrows = [
                    Arrow(start=[-4, 3.5, 0], end=[2, 3.5, 0], color=YELLOW),
                    Arrow(start=[-4, 2.5, 0], end=[2, 2.5, 0], tip_shape=ArrowTriangleTip),
                    Arrow(start=[-4, 1.5, 0], end=[2, 1.5, 0]),
                    Arrow(start=[-4, 0.5, 0], end=[2, 0.5, 0], tip_shape=ArrowSquareTip),

                    Arrow([-4, -0.5, 0], [2, -0.5, 0], tip_shape=ArrowSquareFilledTip),
                    Arrow([-4, -1.5, 0], [2, -1.5, 0], tip_shape=ArrowCircleTip),
                    Arrow([-4, -2.5, 0], [2, -2.5, 0], tip_shape=ArrowCircleFilledTip),
                    Arrow([-4, -3.5, 0], [2, -3.5, 0], tip_shape=StealthTip)
                ]

                small_arrows = (
                    arrow.copy().scale(0.5, scale_tips=True).next_to(arrow, RIGHT) for arrow in big_arrows
                )

                labels = (
                    Text(tip_names[i], font='monospace', font_size=20, color=BLUE).next_to(big_arrows[i], LEFT) for i in range(len(big_arrows))
                )

                self.add(*big_arrows, *small_arrows, *labels)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Has to be implemented in inheriting subclasses.")

    @property
    def base(self) -> Point3D:
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
    def tip_point(self) -> Point3D:
        r"""The tip point of the arrow tip.

        Examples
        --------
        ::

            >>> from manim import Arrow
            >>> arrow = Arrow(np.array([0, 0, 0]), np.array([2, 0, 0]), buff=0)
            >>> arrow.tip.tip_point.round(2) + 0.
            array([2., 0., 0.])

        """
        # Type inference of extracting an element from a list, is not
        # supported by numpy, see this numpy issue
        # https://github.com/numpy/numpy/issues/16544
        tip_point: Point3D = self.points[0]
        return tip_point

    @property
    def vector(self) -> Vector3D:
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
    def tip_angle(self) -> float:
        r"""The angle of the arrow tip.

        Examples
        --------
        ::

            >>> from manim import Arrow
            >>> arrow = Arrow(np.array([0, 0, 0]), np.array([1, 1, 0]), buff=0)
            >>> bool(round(arrow.tip.tip_angle, 5) == round(PI/4, 5))
            True

        """
        return angle_of_vector(self.vector)

    @property
    def length(self) -> float:
        r"""The length of the arrow tip.

        Examples
        --------
        ::

            >>> from manim import Arrow
            >>> arrow = Arrow(np.array([0, 0, 0]), np.array([1, 2, 0]))
            >>> round(arrow.tip.length, 3)
            0.35

        """
        return float(np.linalg.norm(self.vector))


class StealthTip(ArrowTip):
    r"""'Stealth' fighter / kite arrow shape.

    Naming is inspired by the corresponding
    `TikZ arrow shape <https://tikz.dev/tikz-arrows#sec-16.3>`__.
    """

    def __init__(
        self,
        fill_opacity: float = 1,
        stroke_width: float = 3,
        length: float = DEFAULT_ARROW_TIP_LENGTH / 2,
        start_angle: float = PI,
        **kwargs: Any,
    ):
        self.start_angle = start_angle
        VMobject.__init__(
            self, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )
        self.set_points_as_corners(
            np.array(
                [
                    [2, 0, 0],  # tip
                    [-1.2, 1.6, 0],
                    [0, 0, 0],  # base
                    [-1.2, -1.6, 0],
                    [2, 0, 0],  # close path, back to tip
                ]
            )
        )
        self.scale(length / self.length)

    @property
    def length(self) -> float:
        """The length of the arrow tip.

        In this case, the length is computed as the height of
        the triangle encompassing the stealth tip (otherwise,
        the tip is scaled too large).
        """
        return float(np.linalg.norm(self.vector) * 1.6)


class ArrowTriangleTip(ArrowTip, Triangle):
    r"""Triangular arrow tip."""

    def __init__(
        self,
        fill_opacity: float = 0,
        stroke_width: float = 3,
        length: float = DEFAULT_ARROW_TIP_LENGTH,
        width: float = DEFAULT_ARROW_TIP_LENGTH,
        start_angle: float = PI,
        **kwargs: Any,
    ) -> None:
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

    def __init__(
        self, fill_opacity: float = 1, stroke_width: float = 0, **kwargs: Any
    ) -> None:
        super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)


class ArrowCircleTip(ArrowTip, Circle):
    r"""Circular arrow tip."""

    def __init__(
        self,
        fill_opacity: float = 0,
        stroke_width: float = 3,
        length: float = DEFAULT_ARROW_TIP_LENGTH,
        start_angle: float = PI,
        **kwargs: Any,
    ) -> None:
        self.start_angle = start_angle
        Circle.__init__(
            self, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )
        self.width = length
        self.stretch_to_fit_height(length)


class ArrowCircleFilledTip(ArrowCircleTip):
    r"""Circular arrow tip with filled tip."""

    def __init__(
        self, fill_opacity: float = 1, stroke_width: float = 0, **kwargs: Any
    ) -> None:
        super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)


class ArrowSquareTip(ArrowTip, Square):
    r"""Square arrow tip."""

    def __init__(
        self,
        fill_opacity: float = 0,
        stroke_width: float = 3,
        length: float = DEFAULT_ARROW_TIP_LENGTH,
        start_angle: float = PI,
        **kwargs: Any,
    ) -> None:
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

    def __init__(
        self, fill_opacity: float = 1, stroke_width: float = 0, **kwargs: Any
    ) -> None:
        super().__init__(fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs)
