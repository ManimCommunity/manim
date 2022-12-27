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
from manim.utils.space_ops import angle_of_vector, cartesian_to_spherical

from ..opengl.opengl_compatibility import ConvertToOpenGL
from ..types.vectorized_mobject import VMobject
from .arc import Circle
from .polygram import Square, Triangle


class TipableVMobject(VMobject, metaclass=ConvertToOpenGL):
    """Meant for shared functionality between Arc and Line.
    Functionality can be classified broadly into these groups:

        * Adding, Creating, Modifying tips
            - add_tip calls create_tip, before pushing the new tip
                into the TipableVMobject's list of submobjects
            - stylistic and positional configuration

        * Checking for tips
            - Boolean checks for whether the TipableVMobject has a tip
                and a starting tip

        * Getters
            - Straightforward accessors, returning information pertaining
                to the TipableVMobject instance's tip(s), its length etc
    """

    def __init__(
        self,
        tip_length=DEFAULT_ARROW_TIP_LENGTH,
        normal_vector=OUT,
        tip_style={},
        **kwargs,
    ):
        self.tip_length = tip_length
        self.normal_vector = normal_vector
        self.tip_style = tip_style
        super().__init__(**kwargs)

    # Adding, Creating, Modifying tips

    def add_tip(
        self, tip=None, tip_shape=None, tip_length=None, tip_width=None, at_start=False
    ):
        """Adds a tip to the TipableVMobject instance, recognising
        that the endpoints might need to be switched if it's
        a 'starting tip' or not.
        """
        if tip is None:
            tip = self.create_tip(tip_shape, tip_length, tip_width, at_start)
        else:
            self.position_tip(tip, at_start)
        self.reset_endpoints_based_on_tip(tip, at_start)
        self.asign_tip_attr(tip, at_start)
        self.add(tip)
        return self

    def create_tip(
        self, tip_shape=None, tip_length=None, tip_width=None, at_start=False
    ):
        """Stylises the tip, positions it spatially, and returns
        the newly instantiated tip to the caller.
        """
        tip = self.get_unpositioned_tip(tip_shape, tip_length, tip_width)
        self.position_tip(tip, at_start)
        return tip

    def get_unpositioned_tip(self, tip_shape=None, tip_length=None, tip_width=None):
        """Returns a tip that has been stylistically configured,
        but has not yet been given a position in space.
        """
        from manim.mobject.geometry.tips import ArrowTriangleFilledTip

        style = {}

        if tip_shape is None:
            tip_shape = ArrowTriangleFilledTip

        if tip_shape is ArrowTriangleFilledTip:
            if tip_width is None:
                tip_width = self.get_default_tip_length()
            style.update({"width": tip_width})
        if tip_length is None:
            tip_length = self.get_default_tip_length()

        color = self.get_color()
        style.update({"fill_color": color, "stroke_color": color})
        style.update(self.tip_style)
        tip = tip_shape(length=tip_length, **style)
        return tip

    def position_tip(self, tip, at_start=False):
        # Last two control points, defining both
        # the end, and the tangency direction
        if at_start:
            anchor = self.get_start()
            handle = self.get_first_handle()
        else:
            handle = self.get_last_handle()
            anchor = self.get_end()
        angles = cartesian_to_spherical(handle - anchor)
        tip.rotate(
            angles[1] - PI - tip.tip_angle,
        )  # Rotates the tip along the azimuthal
        if not hasattr(self, "_init_positioning_axis"):
            axis = [
                np.sin(angles[1]),
                -np.cos(angles[1]),
                0,
            ]  # Obtains the perpendicular of the tip
            tip.rotate(
                -angles[2] + PI / 2,
                axis=axis,
            )  # Rotates the tip along the vertical wrt the axis
            self._init_positioning_axis = axis
        tip.shift(anchor - tip.tip_point)
        return tip

    def reset_endpoints_based_on_tip(self, tip, at_start):
        if self.get_length() == 0:
            # Zero length, put_start_and_end_on wouldn't work
            return self

        if at_start:
            self.put_start_and_end_on(tip.base, self.get_end())
        else:
            self.put_start_and_end_on(self.get_start(), tip.base)
        return self

    def asign_tip_attr(self, tip, at_start):
        if at_start:
            self.start_tip = tip
        else:
            self.tip = tip
        return self

    # Checking for tips

    def has_tip(self):
        return hasattr(self, "tip") and self.tip in self

    def has_start_tip(self):
        return hasattr(self, "start_tip") and self.start_tip in self

    # Getters

    def pop_tips(self):
        start, end = self.get_start_and_end()
        result = self.get_group_class()()
        if self.has_tip():
            result.add(self.tip)
            self.remove(self.tip)
        if self.has_start_tip():
            result.add(self.start_tip)
            self.remove(self.start_tip)
        self.put_start_and_end_on(start, end)
        return result

    def get_tips(self):
        """Returns a VGroup (collection of VMobjects) containing
        the TipableVMObject instance's tips.
        """
        result = self.get_group_class()()
        if hasattr(self, "tip"):
            result.add(self.tip)
        if hasattr(self, "start_tip"):
            result.add(self.start_tip)
        return result

    def get_tip(self):
        """Returns the TipableVMobject instance's (first) tip,
        otherwise throws an exception."""
        tips = self.get_tips()
        if len(tips) == 0:
            raise Exception("tip not found")
        else:
            return tips[0]

    def get_default_tip_length(self):
        return self.tip_length

    def get_first_handle(self):
        return self.points[1]

    def get_last_handle(self):
        return self.points[-2]

    def get_end(self):
        if self.has_tip():
            return self.tip.get_start()
        else:
            return super().get_end()

    def get_start(self):
        if self.has_start_tip():
            return self.start_tip.get_start()
        else:
            return super().get_start()

    def get_length(self):
        start, end = self.get_start_and_end()
        return np.linalg.norm(start - end)


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
