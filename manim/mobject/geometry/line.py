r"""Mobjects that are lines or variations of them."""

from __future__ import annotations

__all__ = [
    "Line",
    "DashedLine",
    "TangentLine",
    "Elbow",
    "Arrow",
    "Vector",
    "DoubleArrow",
    "Angle",
    "RightAngle",
]

from typing import Any, Sequence

import numpy as np
from colour import Color

from manim import config
from manim.constants import *
from manim.mobject.geometry.arc import Arc, ArcBetweenPoints, Dot, TipableVMobject
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from manim.mobject.mobject import Mobject
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.types.vectorized_mobject import DashedVMobject, VGroup, VMobject
from manim.utils.color import *
from manim.utils.color import Colors
from manim.utils.space_ops import angle_of_vector, line_intersection, normalize


class Line(TipableVMobject):
    def __init__(self, start=LEFT, end=RIGHT, buff=0, path_arc=None, **kwargs):
        self.dim = 3
        self.buff = buff
        self.path_arc = path_arc
        self._set_start_and_end_attrs(start, end)
        super().__init__(**kwargs)

    def generate_points(self):
        self.set_points_by_ends(
            start=self.start,
            end=self.end,
            buff=self.buff,
            path_arc=self.path_arc,
        )

    def set_points_by_ends(self, start, end, buff=0, path_arc=0):
        if path_arc:
            arc = ArcBetweenPoints(self.start, self.end, angle=self.path_arc)
            self.set_points(arc.points)
        else:
            self.set_points_as_corners([start, end])

        self._account_for_buff(buff)

    init_points = generate_points

    def _account_for_buff(self, buff):
        if buff == 0:
            return
        #
        if self.path_arc == 0:
            length = self.get_length()
        else:
            length = self.get_arc_length()
        #
        if length < 2 * buff:
            return
        buff_proportion = buff / length
        self.pointwise_become_partial(self, buff_proportion, 1 - buff_proportion)
        return self

    def _set_start_and_end_attrs(self, start, end):
        # If either start or end are Mobjects, this
        # gives their centers
        rough_start = self._pointify(start)
        rough_end = self._pointify(end)
        vect = normalize(rough_end - rough_start)
        # Now that we know the direction between them,
        # we can find the appropriate boundary point from
        # start and end, if they're mobjects
        self.start = self._pointify(start, vect)
        self.end = self._pointify(end, -vect)

    def _pointify(
        self,
        mob_or_point: Mobject | Sequence[float],
        direction: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Transforms a mobject into its corresponding point. Does nothing if a point is passed.

        ``direction`` determines the location of the point along its bounding box in that direction.

        Parameters
        ----------
        mob_or_point
            The mobject or point.
        direction
            The direction.
        """
        if isinstance(mob_or_point, (Mobject, OpenGLMobject)):
            mob = mob_or_point
            if direction is None:
                return mob.get_center()
            else:
                return mob.get_boundary_point(direction)
        return np.array(mob_or_point)

    def set_path_arc(self, new_value):
        self.path_arc = new_value
        self.init_points()

    def put_start_and_end_on(self, start: Sequence[float], end: Sequence[float]):
        """Sets starts and end coordinates of a line.

        Examples
        --------
        .. manim:: LineExample

            class LineExample(Scene):
                def construct(self):
                    d = VGroup()
                    for i in range(0,10):
                        d.add(Dot())
                    d.arrange_in_grid(buff=1)
                    self.add(d)
                    l= Line(d[0], d[1])
                    self.add(l)
                    self.wait()
                    l.put_start_and_end_on(d[1].get_center(), d[2].get_center())
                    self.wait()
                    l.put_start_and_end_on(d[4].get_center(), d[7].get_center())
                    self.wait()
        """
        curr_start, curr_end = self.get_start_and_end()
        if np.all(curr_start == curr_end):
            # TODO, any problems with resetting
            # these attrs?
            self.start = start
            self.end = end
            self.generate_points()
        return super().put_start_and_end_on(start, end)

    def get_vector(self):
        return self.get_end() - self.get_start()

    def get_unit_vector(self):
        return normalize(self.get_vector())

    def get_angle(self):
        return angle_of_vector(self.get_vector())

    def get_projection(self, point: Sequence[float]) -> Sequence[float]:
        """Returns the projection of a point onto a line.

        Parameters
        ----------
        point
            The point to which the line is projected.
        """

        start = self.get_start()
        end = self.get_end()
        unit_vect = normalize(end - start)
        return start + np.dot(point - start, unit_vect) * unit_vect

    def get_slope(self):
        return np.tan(self.get_angle())

    def set_angle(self, angle, about_point=None):
        if about_point is None:
            about_point = self.get_start()

        self.rotate(
            angle - self.get_angle(),
            about_point=about_point,
        )

        return self

    def set_length(self, length):
        return self.scale(length / self.get_length())


class DashedLine(Line):
    """A dashed :class:`Line`.

    Parameters
    ----------
    args
        Arguments to be passed to :class:`Line`
    dash_length
        The length of each individual dash of the line.
    dashed_ratio
        The ratio of dash space to empty space. Range of 0-1.
    kwargs
        Additional arguments to be passed to :class:`Line`


    .. seealso::
        :class:`~.DashedVMobject`

    Examples
    --------
    .. manim:: DashedLineExample
        :save_last_frame:

        class DashedLineExample(Scene):
            def construct(self):
                # dash_length increased
                dashed_1 = DashedLine(config.left_side, config.right_side, dash_length=2.0).shift(UP*2)
                # normal
                dashed_2 = DashedLine(config.left_side, config.right_side)
                # dashed_ratio decreased
                dashed_3 = DashedLine(config.left_side, config.right_side, dashed_ratio=0.1).shift(DOWN*2)
                self.add(dashed_1, dashed_2, dashed_3)
    """

    def __init__(
        self,
        *args: Any,
        dash_length: float = DEFAULT_DASH_LENGTH,
        dashed_ratio: float = 0.5,
        **kwargs,
    ):
        self.dash_length = dash_length
        self.dashed_ratio = dashed_ratio
        super().__init__(*args, **kwargs)
        dashes = DashedVMobject(
            self,
            num_dashes=self._calculate_num_dashes(),
            dashed_ratio=dashed_ratio,
        )
        self.clear_points()
        self.add(*dashes)

    def _calculate_num_dashes(self) -> int:
        """Returns the number of dashes in the dashed line.

        Examples
        --------
        ::

            >>> DashedLine()._calculate_num_dashes()
            20
        """

        # Minimum number of dashes has to be 2
        return max(
            2,
            int(np.ceil((self.get_length() / self.dash_length) * self.dashed_ratio)),
        )

    def get_start(self) -> np.ndarray:
        """Returns the start point of the line.

        Examples
        --------
        ::

            >>> DashedLine().get_start()
            array([-1.,  0.,  0.])
        """

        if len(self.submobjects) > 0:
            return self.submobjects[0].get_start()
        else:
            return super().get_start()

    def get_end(self) -> np.ndarray:
        """Returns the end point of the line.

        Examples
        --------
        ::

            >>> DashedLine().get_end()
            array([1., 0., 0.])
        """

        if len(self.submobjects) > 0:
            return self.submobjects[-1].get_end()
        else:
            return super().get_end()

    def get_first_handle(self) -> np.ndarray:
        """Returns the point of the first handle.

        Examples
        --------
        ::

            >>> DashedLine().get_first_handle()
            array([-0.98333333,  0.        ,  0.        ])
        """

        return self.submobjects[0].points[1]

    def get_last_handle(self) -> np.ndarray:
        """Returns the point of the last handle.

        Examples
        --------
        ::

            >>> DashedLine().get_last_handle()
            array([0.98333333, 0.        , 0.        ])
        """

        return self.submobjects[-1].points[-2]


class TangentLine(Line):
    """Constructs a line tangent to a :class:`~.VMobject` at a specific point.

    Parameters
    ----------
    vmob
        The VMobject on which the tangent line is drawn.
    alpha
        How far along the shape that the line will be constructed. range: 0-1.
    length
        Length of the tangent line.
    d_alpha
        The ``dx`` value
    kwargs
        Additional arguments to be passed to :class:`Line`


    .. seealso::
        :meth:`~.VMobject.point_from_proportion`

    Examples
    --------
    .. manim:: TangentLineExample
        :save_last_frame:

        class TangentLineExample(Scene):
            def construct(self):
                circle = Circle(radius=2)
                line_1 = TangentLine(circle, alpha=0.0, length=4, color=BLUE_D) # right
                line_2 = TangentLine(circle, alpha=0.4, length=4, color=GREEN) # top left
                self.add(circle, line_1, line_2)
    """

    def __init__(
        self,
        vmob: VMobject,
        alpha: float,
        length: float = 1,
        d_alpha: float = 1e-6,
        **kwargs,
    ):
        self.length = length
        self.d_alpha = d_alpha
        da = self.d_alpha
        a1 = np.clip(alpha - da, 0, 1)
        a2 = np.clip(alpha + da, 0, 1)
        super().__init__(
            vmob.point_from_proportion(a1), vmob.point_from_proportion(a2), **kwargs
        )
        self.scale(self.length / self.get_length())


class Elbow(VMobject, metaclass=ConvertToOpenGL):
    """Two lines that create a right angle about each other: L-shape.

    Parameters
    ----------
    width
        The length of the elbow's sides.
    angle
        The rotation of the elbow.
    kwargs
        Additional arguments to be passed to :class:`~.VMobject`

    .. seealso::
        :class:`RightAngle`

    Examples
    --------
    .. manim:: ElbowExample
        :save_last_frame:

        class ElbowExample(Scene):
            def construct(self):
                elbow_1 = Elbow()
                elbow_2 = Elbow(width=2.0)
                elbow_3 = Elbow(width=2.0, angle=5*PI/4)

                elbow_group = Group(elbow_1, elbow_2, elbow_3).arrange(buff=1)
                self.add(elbow_group)
    """

    def __init__(self, width: float = 0.2, angle: float = 0, **kwargs):
        self.angle = angle
        super().__init__(**kwargs)
        self.set_points_as_corners([UP, UP + RIGHT, RIGHT])
        self.scale_to_fit_width(width, about_point=ORIGIN)
        self.rotate(self.angle, about_point=ORIGIN)


class Arrow(Line):
    """An arrow.

    Parameters
    ----------
    args
        Arguments to be passed to :class:`Line`.
    stroke_width
        The thickness of the arrow. Influenced by :attr:`max_stroke_width_to_length_ratio`.
    buff
        The distance of the arrow from its start and end points.
    max_tip_length_to_length_ratio
        :attr:`tip_length` scales with the length of the arrow. Increasing this ratio raises the max value of :attr:`tip_length`.
    max_stroke_width_to_length_ratio
        :attr:`stroke_width` scales with the length of the arrow. Increasing this ratio ratios the max value of :attr:`stroke_width`.
    kwargs
        Additional arguments to be passed to :class:`Line`.


    .. seealso::
        :class:`ArrowTip`
        :class:`CurvedArrow`

    Examples
    --------
    .. manim:: ArrowExample
        :save_last_frame:

        from manim.mobject.geometry.tips import ArrowSquareTip
        class ArrowExample(Scene):
            def construct(self):
                arrow_1 = Arrow(start=RIGHT, end=LEFT, color=GOLD)
                arrow_2 = Arrow(start=RIGHT, end=LEFT, color=GOLD, tip_shape=ArrowSquareTip).shift(DOWN)
                g1 = Group(arrow_1, arrow_2)

                # the effect of buff
                square = Square(color=MAROON_A)
                arrow_3 = Arrow(start=LEFT, end=RIGHT)
                arrow_4 = Arrow(start=LEFT, end=RIGHT, buff=0).next_to(arrow_1, UP)
                g2 = Group(arrow_3, arrow_4, square)

                # a shorter arrow has a shorter tip and smaller stroke width
                arrow_5 = Arrow(start=ORIGIN, end=config.top).shift(LEFT * 4)
                arrow_6 = Arrow(start=config.top + DOWN, end=config.top).shift(LEFT * 3)
                g3 = Group(arrow_5, arrow_6)

                self.add(Group(g1, g2, g3).arrange(buff=2))


    .. manim:: ArrowExample
        :save_last_frame:

        class ArrowExample(Scene):
            def construct(self):
                left_group = VGroup()
                # As buff increases, the size of the arrow decreases.
                for buff in np.arange(0, 2.2, 0.45):
                    left_group += Arrow(buff=buff, start=2 * LEFT, end=2 * RIGHT)
                # Required to arrange arrows.
                left_group.arrange(DOWN)
                left_group.move_to(4 * LEFT)

                middle_group = VGroup()
                # As max_stroke_width_to_length_ratio gets bigger,
                # the width of stroke increases.
                for i in np.arange(0, 5, 0.5):
                    middle_group += Arrow(max_stroke_width_to_length_ratio=i)
                middle_group.arrange(DOWN)

                UR_group = VGroup()
                # As max_tip_length_to_length_ratio increases,
                # the length of the tip increases.
                for i in np.arange(0, 0.3, 0.1):
                    UR_group += Arrow(max_tip_length_to_length_ratio=i)
                UR_group.arrange(DOWN)
                UR_group.move_to(4 * RIGHT + 2 * UP)

                DR_group = VGroup()
                DR_group += Arrow(start=LEFT, end=RIGHT, color=BLUE, tip_shape=ArrowSquareTip)
                DR_group += Arrow(start=LEFT, end=RIGHT, color=BLUE, tip_shape=ArrowSquareFilledTip)
                DR_group += Arrow(start=LEFT, end=RIGHT, color=YELLOW, tip_shape=ArrowCircleTip)
                DR_group += Arrow(start=LEFT, end=RIGHT, color=YELLOW, tip_shape=ArrowCircleFilledTip)
                DR_group.arrange(DOWN)
                DR_group.move_to(4 * RIGHT + 2 * DOWN)

                self.add(left_group, middle_group, UR_group, DR_group)
    """

    def __init__(
        self,
        *args: Any,
        stroke_width: float = 6,
        buff: float = MED_SMALL_BUFF,
        max_tip_length_to_length_ratio: float = 0.25,
        max_stroke_width_to_length_ratio: float = 5,
        **kwargs,
    ):
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_stroke_width_to_length_ratio = max_stroke_width_to_length_ratio
        tip_shape = kwargs.pop("tip_shape", ArrowTriangleFilledTip)
        super().__init__(*args, buff=buff, stroke_width=stroke_width, **kwargs)
        # TODO, should this be affected when
        # Arrow.set_stroke is called?
        self.initial_stroke_width = self.stroke_width
        self.add_tip(tip_shape=tip_shape)
        self._set_stroke_width_from_length()

    def scale(self, factor, scale_tips=False, **kwargs):
        r"""Scale an arrow, but keep stroke width and arrow tip size fixed.


        .. seealso::
            :meth:`~.Mobject.scale`

        Examples
        --------
        ::

            >>> arrow = Arrow(np.array([-1, -1, 0]), np.array([1, 1, 0]), buff=0)
            >>> scaled_arrow = arrow.scale(2)
            >>> np.round(scaled_arrow.get_start_and_end(), 8) + 0
            array([[-2., -2.,  0.],
                   [ 2.,  2.,  0.]])
            >>> arrow.tip.length == scaled_arrow.tip.length
            True

        Manually scaling the object using the default method
        :meth:`~.Mobject.scale` does not have the same properties::

            >>> new_arrow = Arrow(np.array([-1, -1, 0]), np.array([1, 1, 0]), buff=0)
            >>> another_scaled_arrow = VMobject.scale(new_arrow, 2)
            >>> another_scaled_arrow.tip.length == arrow.tip.length
            False

        """
        if self.get_length() == 0:
            return self

        if scale_tips:
            super().scale(factor, **kwargs)
            self._set_stroke_width_from_length()
            return self

        has_tip = self.has_tip()
        has_start_tip = self.has_start_tip()
        if has_tip or has_start_tip:
            old_tips = self.pop_tips()

        super().scale(factor, **kwargs)
        self._set_stroke_width_from_length()

        if has_tip:
            self.add_tip(tip=old_tips[0])
        if has_start_tip:
            self.add_tip(tip=old_tips[1], at_start=True)
        return self

    def get_normal_vector(self) -> np.ndarray:
        """Returns the normal of a vector.

        Examples
        --------
        ::

            >>> np.round(Arrow().get_normal_vector()) + 0. # add 0. to avoid negative 0 in output
            array([ 0.,  0., -1.])
        """

        p0, p1, p2 = self.tip.get_start_anchors()[:3]
        return normalize(np.cross(p2 - p1, p1 - p0))

    def reset_normal_vector(self):
        """Resets the normal of a vector"""
        self.normal_vector = self.get_normal_vector()
        return self

    def get_default_tip_length(self) -> float:
        """Returns the default tip_length of the arrow.

        Examples
        --------

        ::

            >>> Arrow().get_default_tip_length()
            0.35
        """

        max_ratio = self.max_tip_length_to_length_ratio
        return min(self.tip_length, max_ratio * self.get_length())

    def _set_stroke_width_from_length(self):
        """Sets stroke width based on length."""
        max_ratio = self.max_stroke_width_to_length_ratio
        if config.renderer == RendererType.OPENGL:
            self.set_stroke(
                width=min(self.initial_stroke_width, max_ratio * self.get_length()),
                recurse=False,
            )
        else:
            self.set_stroke(
                width=min(self.initial_stroke_width, max_ratio * self.get_length()),
                family=False,
            )
        return self


class Vector(Arrow):
    """A vector specialized for use in graphs.

    Parameters
    ----------
    direction
        The direction of the arrow.
    buff
         The distance of the vector from its endpoints.
    kwargs
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

    def __init__(self, direction: list | np.ndarray = RIGHT, buff: float = 0, **kwargs):
        self.buff = buff
        if len(direction) == 2:
            direction = np.hstack([direction, 0])

        super().__init__(ORIGIN, direction, buff=buff, **kwargs)

    def coordinate_label(
        self,
        integer_labels: bool = True,
        n_dim: int = 2,
        color: Color | None = None,
        **kwargs,
    ):
        """Creates a label based on the coordinates of the vector.

        Parameters
        ----------
        integer_labels
            Whether or not to round the coordinates to integers.
        n_dim
            The number of dimensions of the vector.
        color
            Sets the color of label, optional.
        kwargs
            Additional arguments to be passed to :class:`~.Matrix`.

        Returns
        -------
        :class:`~.Matrix`
            The label.

        Examples
        --------
        .. manim:: VectorCoordinateLabel
            :save_last_frame:

            class VectorCoordinateLabel(Scene):
                def construct(self):
                    plane = NumberPlane()

                    vec_1 = Vector([1, 2])
                    vec_2 = Vector([-3, -2])
                    label_1 = vec_1.coordinate_label()
                    label_2 = vec_2.coordinate_label(color=YELLOW)

                    self.add(plane, vec_1, vec_2, label_1, label_2)
        """

        # avoiding circular imports
        from ..matrix import Matrix

        vect = np.array(self.get_end())
        if integer_labels:
            vect = np.round(vect).astype(int)
        vect = vect[:n_dim]
        vect = vect.reshape((n_dim, 1))
        label = Matrix(vect, **kwargs)
        label.scale(LARGE_BUFF - 0.2)

        shift_dir = np.array(self.get_end())
        if shift_dir[0] >= 0:  # Pointing right
            shift_dir -= label.get_left() + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * LEFT
        else:  # Pointing left
            shift_dir -= label.get_right() + DEFAULT_MOBJECT_TO_MOBJECT_BUFFER * RIGHT
        label.shift(shift_dir)
        if color is not None:
            label.set_color(color)
        return label


class DoubleArrow(Arrow):
    """An arrow with tips on both ends.

    Parameters
    ----------
    args
        Arguments to be passed to :class:`Arrow`
    kwargs
        Additional arguments to be passed to :class:`Arrow`


    .. seealso::
        :class:`.~ArrowTip`
        :class:`.~CurvedDoubleArrow`

    Examples
    --------
    .. manim:: DoubleArrowExample
        :save_last_frame:

        from manim.mobject.geometry.tips import ArrowCircleFilledTip
        class DoubleArrowExample(Scene):
            def construct(self):
                circle = Circle(radius=2.0)
                d_arrow = DoubleArrow(start=circle.get_left(), end=circle.get_right())
                d_arrow_2 = DoubleArrow(tip_shape_end=ArrowCircleFilledTip, tip_shape_start=ArrowCircleFilledTip)
                group = Group(Group(circle, d_arrow), d_arrow_2).arrange(UP, buff=1)
                self.add(group)


    .. manim:: DoubleArrowExample2
        :save_last_frame:

        class DoubleArrowExample2(Scene):
            def construct(self):
                box = Square()
                p1 = box.get_left()
                p2 = box.get_right()
                d1 = DoubleArrow(p1, p2, buff=0)
                d2 = DoubleArrow(p1, p2, buff=0, tip_length=0.2, color=YELLOW)
                d3 = DoubleArrow(p1, p2, buff=0, tip_length=0.4, color=BLUE)
                Group(d1, d2, d3).arrange(DOWN)
                self.add(box, d1, d2, d3)
    """

    def __init__(self, *args: Any, **kwargs):
        if "tip_shape_end" in kwargs:
            kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
        tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
        super().__init__(*args, **kwargs)
        self.add_tip(at_start=True, tip_shape=tip_shape_start)


class Angle(VMobject, metaclass=ConvertToOpenGL):
    """A circular arc or elbow-type mobject representing an angle of two lines.

    Parameters
    ----------
    line1 :
        The first line.
    line2 :
        The second line.
    radius :
        The radius of the :class:`Arc`.
    quadrant
        A sequence of two :class:`int` numbers determining which of the 4 quadrants should be used.
        The first value indicates whether to anchor the arc on the first line closer to the end point (1)
        or start point (-1), and the second value functions similarly for the
        end (1) or start (-1) of the second line.
        Possibilities: (1,1), (-1,1), (1,-1), (-1,-1).
    other_angle :
        Toggles between the two possible angles defined by two points and an arc center. If set to
        False (default), the arc will always go counterclockwise from the point on line1 until
        the point on line2 is reached. If set to True, the angle will go clockwise from line1 to line2.
    dot
        Allows for a :class:`Dot` in the arc. Mainly used as an convention to indicate a right angle.
        The dot can be customized in the next three parameters.
    dot_radius
        The radius of the :class:`Dot`. If not specified otherwise, this radius will be 1/10 of the arc radius.
    dot_distance
        Relative distance from the center to the arc: 0 puts the dot in the center and 1 on the arc itself.
    dot_color
        The color of the :class:`Dot`.
    elbow
        Produces an elbow-type mobject indicating a right angle, see :class:`RightAngle` for more information
        and a shorthand.
    **kwargs
        Further keyword arguments that are passed to the constructor of :class:`Arc` or :class:`Elbow`.

    Examples
    --------
    The first example shows some right angles with a dot in the middle while the second example shows
    all 8 possible angles defined by two lines.

    .. manim:: RightArcAngleExample
        :save_last_frame:

        class RightArcAngleExample(Scene):
            def construct(self):
                line1 = Line( LEFT, RIGHT )
                line2 = Line( DOWN, UP )
                rightarcangles = [
                    Angle(line1, line2, dot=True),
                    Angle(line1, line2, radius=0.4, quadrant=(1,-1), dot=True, other_angle=True),
                    Angle(line1, line2, radius=0.5, quadrant=(-1,1), stroke_width=8, dot=True, dot_color=YELLOW, dot_radius=0.04, other_angle=True),
                    Angle(line1, line2, radius=0.7, quadrant=(-1,-1), color=RED, dot=True, dot_color=GREEN, dot_radius=0.08),
                ]
                plots = VGroup()
                for angle in rightarcangles:
                    plot=VGroup(line1.copy(),line2.copy(), angle)
                    plots.add(plot)
                plots.arrange(buff=1.5)
                self.add(plots)

    .. manim:: AngleExample
        :save_last_frame:

        class AngleExample(Scene):
            def construct(self):
                line1 = Line( LEFT + (1/3) * UP, RIGHT + (1/3) * DOWN )
                line2 = Line( DOWN + (1/3) * RIGHT, UP + (1/3) * LEFT )
                angles = [
                    Angle(line1, line2),
                    Angle(line1, line2, radius=0.4, quadrant=(1,-1), other_angle=True),
                    Angle(line1, line2, radius=0.5, quadrant=(-1,1), stroke_width=8, other_angle=True),
                    Angle(line1, line2, radius=0.7, quadrant=(-1,-1), color=RED),
                    Angle(line1, line2, other_angle=True),
                    Angle(line1, line2, radius=0.4, quadrant=(1,-1)),
                    Angle(line1, line2, radius=0.5, quadrant=(-1,1), stroke_width=8),
                    Angle(line1, line2, radius=0.7, quadrant=(-1,-1), color=RED, other_angle=True),
                ]
                plots = VGroup()
                for angle in angles:
                    plot=VGroup(line1.copy(),line2.copy(), angle)
                    plots.add(VGroup(plot,SurroundingRectangle(plot, buff=0.3)))
                plots.arrange_in_grid(rows=2,buff=1)
                self.add(plots)

    .. manim:: FilledAngle
        :save_last_frame:

        class FilledAngle(Scene):
            def construct(self):
                l1 = Line(ORIGIN, 2 * UP + RIGHT).set_color(GREEN)
                l2 = (
                    Line(ORIGIN, 2 * UP + RIGHT)
                    .set_color(GREEN)
                    .rotate(-20 * DEGREES, about_point=ORIGIN)
                )
                norm = l1.get_length()
                a1 = Angle(l1, l2, other_angle=True, radius=norm - 0.5).set_color(GREEN)
                a2 = Angle(l1, l2, other_angle=True, radius=norm).set_color(GREEN)
                q1 = a1.points #  save all coordinates of points of angle a1
                q2 = a2.reverse_direction().points  #  save all coordinates of points of angle a1 (in reversed direction)
                pnts = np.concatenate([q1, q2, q1[0].reshape(1, 3)])  # adds points and ensures that path starts and ends at same point
                mfill = VMobject().set_color(ORANGE)
                mfill.set_points_as_corners(pnts).set_fill(GREEN, opacity=1)
                self.add(l1, l2)
                self.add(mfill)

    """

    def __init__(
        self,
        line1: Line,
        line2: Line,
        radius: float = None,
        quadrant: Sequence[int] = (1, 1),
        other_angle: bool = False,
        dot: bool = False,
        dot_radius: float | None = None,
        dot_distance: float = 0.55,
        dot_color: Colors = WHITE,
        elbow: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lines = (line1, line2)
        self.quadrant = quadrant
        self.dot_distance = dot_distance
        self.elbow = elbow
        inter = line_intersection(
            [line1.get_start(), line1.get_end()],
            [line2.get_start(), line2.get_end()],
        )

        if radius is None:
            if quadrant[0] == 1:
                dist_1 = np.linalg.norm(line1.get_end() - inter)
            else:
                dist_1 = np.linalg.norm(line1.get_start() - inter)
            if quadrant[1] == 1:
                dist_2 = np.linalg.norm(line2.get_end() - inter)
            else:
                dist_2 = np.linalg.norm(line2.get_start() - inter)
            if np.minimum(dist_1, dist_2) < 0.6:
                radius = (2 / 3) * np.minimum(dist_1, dist_2)
            else:
                radius = 0.4
        else:
            self.radius = radius

        anchor_angle_1 = inter + quadrant[0] * radius * line1.get_unit_vector()
        anchor_angle_2 = inter + quadrant[1] * radius * line2.get_unit_vector()

        if elbow:
            anchor_middle = (
                inter
                + quadrant[0] * radius * line1.get_unit_vector()
                + quadrant[1] * radius * line2.get_unit_vector()
            )
            angle_mobject = Elbow(**kwargs)
            angle_mobject.set_points_as_corners(
                [anchor_angle_1, anchor_middle, anchor_angle_2],
            )
        else:
            angle_1 = angle_of_vector(anchor_angle_1 - inter)
            angle_2 = angle_of_vector(anchor_angle_2 - inter)

            if not other_angle:
                start_angle = angle_1
                if angle_2 > angle_1:
                    angle_fin = angle_2 - angle_1
                else:
                    angle_fin = 2 * np.pi - (angle_1 - angle_2)
            else:
                start_angle = angle_1
                if angle_2 < angle_1:
                    angle_fin = -angle_1 + angle_2
                else:
                    angle_fin = -2 * np.pi + (angle_2 - angle_1)

            self.angle_value = angle_fin

            angle_mobject = Arc(
                radius=radius,
                angle=self.angle_value,
                start_angle=start_angle,
                arc_center=inter,
                **kwargs,
            )

            if dot:
                if dot_radius is None:
                    dot_radius = radius / 10
                else:
                    self.dot_radius = dot_radius
                right_dot = Dot(ORIGIN, radius=dot_radius, color=dot_color)
                dot_anchor = (
                    inter
                    + (angle_mobject.get_center() - inter)
                    / np.linalg.norm(angle_mobject.get_center() - inter)
                    * radius
                    * dot_distance
                )
                right_dot.move_to(dot_anchor)
                self.add(right_dot)

        self.set_points(angle_mobject.points)

    def get_lines(self) -> VGroup:
        """Get the lines forming an angle of the :class:`Angle` class.

        Returns
        -------
        :class:`~.VGroup`
            A :class:`~.VGroup` containing the lines that form the angle of the :class:`Angle` class.

        Examples
        --------
        ::

            >>> line_1, line_2 = Line(ORIGIN, RIGHT), Line(ORIGIN, UR)
            >>> angle = Angle(line_1, line_2)
            >>> angle.get_lines()
            VGroup(Line, Line)
        """

        return VGroup(*self.lines)

    def get_value(self, degrees: bool = False) -> float:
        """Get the value of an angle of the :class:`Angle` class.

        Parameters
        ----------
        degrees
            A boolean to decide the unit (deg/rad) in which the value of the angle is returned.

        Returns
        -------
        :class:`float`
            The value in degrees/radians of an angle of the :class:`Angle` class.

        Examples
        --------

        .. manim:: GetValueExample
            :save_last_frame:

            class GetValueExample(Scene):
                def construct(self):
                    line1 = Line(LEFT+(1/3)*UP, RIGHT+(1/3)*DOWN)
                    line2 = Line(DOWN+(1/3)*RIGHT, UP+(1/3)*LEFT)

                    angle = Angle(line1, line2, radius=0.4)

                    value = DecimalNumber(angle.get_value(degrees=True), unit="^{\\circ}")
                    value.next_to(angle, UR)

                    self.add(line1, line2, angle, value)
        """

        if degrees:
            return self.angle_value / DEGREES
        return self.angle_value

    @staticmethod
    def from_three_points(
        A: np.ndarray, B: np.ndarray, C: np.ndarray, **kwargs
    ) -> Angle:
        """The angle between the lines AB and BC.

        This constructs the angle :math:`\\angle ABC`.

        Parameters
        ----------
        A
            The endpoint of the first angle leg
        B
            The vertex of the angle
        C
            The endpoint of the second angle leg

        **kwargs
            Further keyword arguments are passed to :class:`.Angle`

        Returns
        -------
        The Angle calculated from the three points

                    Angle(line1, line2, radius=0.5, quadrant=(-1,1), stroke_width=8),
                    Angle(line1, line2, radius=0.7, quadrant=(-1,-1), color=RED, other_angle=True),

        Examples
        --------
        .. manim:: AngleFromThreePointsExample
            :save_last_frame:

            class AngleFromThreePointsExample(Scene):
                def construct(self):
                    sample_angle = Angle.from_three_points(UP, ORIGIN, LEFT)
                    red_angle = Angle.from_three_points(LEFT + UP, ORIGIN, RIGHT, radius=.8, quadrant=(-1,-1), color=RED, stroke_width=8, other_angle=True)
                    self.add(red_angle, sample_angle)
        """
        return Angle(Line(B, A), Line(B, C), **kwargs)


class RightAngle(Angle):
    """An elbow-type mobject representing a right angle between two lines.

    Parameters
    ----------
    line1
        The first line.
    line2
        The second line.
    length
        The length of the arms.
    **kwargs
        Further keyword arguments that are passed to the constructor of :class:`Angle`.

    Examples
    --------
    .. manim:: RightAngleExample
        :save_last_frame:

        class RightAngleExample(Scene):
            def construct(self):
                line1 = Line( LEFT, RIGHT )
                line2 = Line( DOWN, UP )
                rightangles = [
                    RightAngle(line1, line2),
                    RightAngle(line1, line2, length=0.4, quadrant=(1,-1)),
                    RightAngle(line1, line2, length=0.5, quadrant=(-1,1), stroke_width=8),
                    RightAngle(line1, line2, length=0.7, quadrant=(-1,-1), color=RED),
                ]
                plots = VGroup()
                for rightangle in rightangles:
                    plot=VGroup(line1.copy(),line2.copy(), rightangle)
                    plots.add(plot)
                plots.arrange(buff=1.5)
                self.add(plots)
    """

    def __init__(self, line1: Line, line2: Line, length: float | None = None, **kwargs):
        super().__init__(line1, line2, radius=length, elbow=True, **kwargs)
