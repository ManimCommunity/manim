r"""Mobjects that are simple geometric shapes.

Examples
--------

.. manim:: UsefulAnnotations
    :save_last_frame:

    class UsefulAnnotations(Scene):
        def construct(self):
            m0 = Dot()
            m1 = AnnotationDot()
            m2 = LabeledDot("ii")
            m3 = LabeledDot(MathTex(r"\alpha").set_color(ORANGE))
            m4 = CurvedArrow(2*LEFT, 2*RIGHT, radius= -5)
            m5 = CurvedArrow(2*LEFT, 2*RIGHT, radius= 8)
            m6 = CurvedDoubleArrow(ORIGIN, 2*RIGHT)

            self.add(m0, m1, m2, m3, m4, m5, m6)
            for i, mobj in enumerate(self.mobjects):
                mobj.shift(DOWN * (i-3))

"""

__all__ = [
    "TipableVMobject",
    "Arc",
    "ArcBetweenPoints",
    "CurvedArrow",
    "CurvedDoubleArrow",
    "Circle",
    "Dot",
    "AnnotationDot",
    "LabeledDot",
    "Ellipse",
    "AnnularSector",
    "Sector",
    "Annulus",
    "Line",
    "DashedLine",
    "TangentLine",
    "Elbow",
    "Arrow",
    "Vector",
    "DoubleArrow",
    "CubicBezier",
    "Polygram",
    "Polygon",
    "RegularPolygram",
    "RegularPolygon",
    "Star",
    "ArcPolygon",
    "ArcPolygonFromArcs",
    "Triangle",
    "ArrowTip",
    "Rectangle",
    "Square",
    "RoundedRectangle",
    "Cutout",
    "Angle",
    "RightAngle",
    "ArrowCircleFilledTip",
    "ArrowCircleTip",
    "ArrowSquareTip",
    "ArrowSquareFilledTip",
]

import itertools
import math
import warnings
from typing import Iterable, Optional, Sequence

import numpy as np
from colour import Color

from manim.mobject.opengl_mobject import OpenGLMobject

from .. import config, logger
from ..constants import *
from ..mobject.mobject import Mobject
from ..mobject.types.vectorized_mobject import DashedVMobject, VGroup, VMobject
from ..utils.color import *
from ..utils.deprecation import deprecated_params
from ..utils.iterables import adjacent_n_tuples, adjacent_pairs
from ..utils.space_ops import (
    angle_between_vectors,
    angle_of_vector,
    cartesian_to_spherical,
    line_intersection,
    normalize,
    perpendicular_bisector,
    regular_vertices,
    rotate_vector,
)
from .opengl_compatibility import ConvertToOpenGL


class TipableVMobject(VMobject, metaclass=ConvertToOpenGL):
    """
    Meant for shared functionality between Arc and Line.
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

    def add_tip(self, tip=None, tip_shape=None, tip_length=None, at_start=False):
        """
        Adds a tip to the TipableVMobject instance, recognising
        that the endpoints might need to be switched if it's
        a 'starting tip' or not.
        """
        if tip is None:
            tip = self.create_tip(tip_shape, tip_length, at_start)
        else:
            self.position_tip(tip, at_start)
        self.reset_endpoints_based_on_tip(tip, at_start)
        self.asign_tip_attr(tip, at_start)
        self.add(tip)
        return self

    def create_tip(self, tip_shape=None, tip_length=None, at_start=False):
        """
        Stylises the tip, positions it spatially, and returns
        the newly instantiated tip to the caller.
        """
        tip = self.get_unpositioned_tip(tip_shape, tip_length)
        self.position_tip(tip, at_start)
        return tip

    def get_unpositioned_tip(self, tip_shape=None, tip_length=None):
        """
        Returns a tip that has been stylistically configured,
        but has not yet been given a position in space.
        """
        if tip_shape is None:
            tip_shape = ArrowTriangleFilledTip
        if tip_length is None:
            tip_length = self.get_default_tip_length()
        color = self.get_color()
        style = {"fill_color": color, "stroke_color": color}
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
            angles[2] - PI - tip.tip_angle,
        )  # Rotates the tip along the azimuthal
        if not hasattr(self, "_init_positioning_axis"):
            axis = [
                np.sin(angles[2]),
                -np.cos(angles[2]),
                0,
            ]  # Obtains the perpendicular of the tip
            tip.rotate(
                -angles[1] + PI / 2,
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
        """
        Returns a VGroup (collection of VMobjects) containing
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


class Arc(TipableVMobject):
    """A circular arc.

    Examples
    --------
    A simple arc of angle Pi.

    .. manim:: ArcExample
        :save_last_frame:

        class ArcExample(Scene):
            def construct(self):
                self.add(Arc(angle=PI))
    """

    def __init__(
        self,
        radius: float = 1.0,
        start_angle=0,
        angle=TAU / 4,
        num_components=9,
        arc_center=ORIGIN,
        **kwargs,
    ):
        if radius is None:  # apparently None is passed by ArcBetweenPoints
            radius = 1.0
        self.radius = radius
        self.num_components = num_components
        self.arc_center = arc_center
        self.start_angle = start_angle
        self.angle = angle
        self._failed_to_get_center = False
        super().__init__(**kwargs)

    def generate_points(self):
        self.set_pre_positioned_points()
        self.scale(self.radius, about_point=ORIGIN)
        self.shift(self.arc_center)

    # Points are set a bit differently when rendering via OpenGL.
    # TODO: refactor Arc so that only one strategy for setting points
    # has to be used.
    def init_points(self):
        self.set_points(
            Arc.create_quadratic_bezier_points(
                angle=self.angle,
                start_angle=self.start_angle,
                n_components=self.num_components,
            ),
        )
        self.scale(self.radius, about_point=ORIGIN)
        self.shift(self.arc_center)

    @staticmethod
    def create_quadratic_bezier_points(angle, start_angle=0, n_components=8):
        samples = np.array(
            [
                [np.cos(a), np.sin(a), 0]
                for a in np.linspace(
                    start_angle,
                    start_angle + angle,
                    2 * n_components + 1,
                )
            ],
        )
        theta = angle / n_components
        samples[1::2] /= np.cos(theta / 2)

        points = np.zeros((3 * n_components, 3))
        points[0::3] = samples[0:-1:2]
        points[1::3] = samples[1::2]
        points[2::3] = samples[2::2]
        return points

    def set_pre_positioned_points(self):
        anchors = np.array(
            [
                np.cos(a) * RIGHT + np.sin(a) * UP
                for a in np.linspace(
                    self.start_angle,
                    self.start_angle + self.angle,
                    self.num_components,
                )
            ],
        )
        # Figure out which control points will give the
        # Appropriate tangent lines to the circle
        d_theta = self.angle / (self.num_components - 1.0)
        tangent_vectors = np.zeros(anchors.shape)
        # Rotate all 90 degrees, via (x, y) -> (-y, x)
        tangent_vectors[:, 1] = anchors[:, 0]
        tangent_vectors[:, 0] = -anchors[:, 1]
        # Use tangent vectors to deduce anchors
        handles1 = anchors[:-1] + (d_theta / 3) * tangent_vectors[:-1]
        handles2 = anchors[1:] - (d_theta / 3) * tangent_vectors[1:]
        self.set_anchors_and_handles(anchors[:-1], handles1, handles2, anchors[1:])

    def get_arc_center(self, warning=True):
        """
        Looks at the normals to the first two
        anchors, and finds their intersection points
        """
        # First two anchors and handles
        a1, h1, h2, a2 = self.points[:4]

        if np.all(a1 == a2):
            # For a1 and a2 to lie at the same point arc radius
            # must be zero. Thus arc_center will also lie at
            # that point.
            return a1
        # Tangent vectors
        t1 = h1 - a1
        t2 = h2 - a2
        # Normals
        n1 = rotate_vector(t1, TAU / 4)
        n2 = rotate_vector(t2, TAU / 4)
        try:
            return line_intersection(line1=(a1, a1 + n1), line2=(a2, a2 + n2))
        except Exception:
            if warning:
                warnings.warn("Can't find Arc center, using ORIGIN instead")
            self._failed_to_get_center = True
            return np.array(ORIGIN)

    def move_arc_center_to(self, point):
        self.shift(point - self.get_arc_center())
        return self

    def stop_angle(self):
        return angle_of_vector(self.points[-1] - self.get_arc_center()) % TAU


class ArcBetweenPoints(Arc):
    """
    Inherits from Arc and additionally takes 2 points between which the arc is spanned.

    Example
    --------------------
    .. manim:: ArcBetweenPointsExample

      class ArcBetweenPointsExample(Scene):
          def construct(self):
              circle = Circle(radius=2, stroke_color=GREY)
              dot_1 = Dot(color=GREEN).move_to([2, 0, 0]).scale(0.5)
              dot_1_text = Tex("(2,0)").scale(0.5).next_to(dot_1, RIGHT).set_color(BLUE)
              dot_2 = Dot(color=GREEN).move_to([0, 2, 0]).scale(0.5)
              dot_2_text = Tex("(0,2)").scale(0.5).next_to(dot_2, UP).set_color(BLUE)
              arc= ArcBetweenPoints(start=2 * RIGHT, end=2 * UP, stroke_color=YELLOW)
              self.add(circle, dot_1, dot_2, dot_1_text, dot_2_text)
              self.play(Create(arc))
    """

    def __init__(self, start, end, angle=TAU / 4, radius=None, **kwargs):
        if radius is not None:
            self.radius = radius
            if radius < 0:
                sign = -2
                radius *= -1
            else:
                sign = 2
            halfdist = np.linalg.norm(np.array(start) - np.array(end)) / 2
            if radius < halfdist:
                raise ValueError(
                    """ArcBetweenPoints called with a radius that is
                            smaller than half the distance between the points.""",
                )
            arc_height = radius - math.sqrt(radius ** 2 - halfdist ** 2)
            angle = math.acos((radius - arc_height) / radius) * sign

        super().__init__(radius=radius, angle=angle, **kwargs)
        if angle == 0:
            self.set_points_as_corners([LEFT, RIGHT])
        self.put_start_and_end_on(start, end)

        if radius is None:
            center = self.get_arc_center(warning=False)
            if not self._failed_to_get_center:
                self.radius = np.linalg.norm(np.array(start) - np.array(center))
            else:
                self.radius = math.inf


class CurvedArrow(ArcBetweenPoints):
    def __init__(self, start_point, end_point, **kwargs):
        tip_shape = kwargs.pop("tip_shape", ArrowTriangleFilledTip)
        super().__init__(start_point, end_point, **kwargs)
        self.add_tip(tip_shape=tip_shape)


class CurvedDoubleArrow(CurvedArrow):
    def __init__(self, start_point, end_point, **kwargs):
        if "tip_shape_end" in kwargs:
            kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
        tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
        super().__init__(start_point, end_point, **kwargs)
        self.add_tip(at_start=True, tip_shape=tip_shape_start)


class Circle(Arc):
    """A circle.

    Parameters
    ----------
    color : :class:`~.Colors`, optional
        The color of the shape.
    kwargs : Any
        Additional arguments to be passed to :class:`Arc`

    Examples
    --------

    .. manim:: CircleExample
        :save_last_frame:

        class CircleExample(Scene):
            def construct(self):
                circle_1 = Circle(radius=1.0)
                circle_2 = Circle(radius=1.5, color=GREEN)
                circle_3 = Circle(radius=1.0, color=BLUE_B, fill_opacity=1)

                circle_group = Group(circle_1, circle_2, circle_3).arrange(buff=1)
                self.add(circle_group)
    """

    def __init__(
        self,
        radius: float = None,
        color=RED,
        **kwargs,
    ):
        super().__init__(
            radius=radius,
            start_angle=0,
            angle=TAU,
            color=color,
            **kwargs,
        )

    def surround(self, mobject, dim_to_match=0, stretch=False, buffer_factor=1.2):
        """Modifies a circle so that it surrounds a given mobject.

        Parameters
        ----------
        mobject : :class:`~.Mobject`
            The mobject that the circle will be surrounding.
        dim_to_match : :class:`int`, optional
        buffer_factor :  :class:`float`, optional
            Scales the circle with respect to the mobject. A `buffer_factor` < 1 makes the circle smaller than the mobject.
        stretch : :class:`bool`, optional
            Stretches the circle to fit more tightly around the mobject. Note: Does not work with :class:`Line`

        Examples
        --------

        .. manim:: CircleSurround
            :save_last_frame:

            class CircleSurround(Scene):
                def construct(self):
                    triangle1 = Triangle()
                    circle1 = Circle().surround(triangle1)
                    group1 = Group(triangle1,circle1) # treat the two mobjects as one

                    line2 = Line()
                    circle2 = Circle().surround(line2, buffer_factor=2.0)
                    group2 = Group(line2,circle2)

                    # buffer_factor < 1, so the circle is smaller than the square
                    square3 = Square()
                    circle3 = Circle().surround(square3, buffer_factor=0.5)
                    group3 = Group(square3, circle3)

                    group = Group(group1, group2, group3).arrange(buff=1)
                    self.add(group)
        """

        # Ignores dim_to_match and stretch; result will always be a circle
        # TODO: Perhaps create an ellipse class to handle single-dimension stretching

        # Something goes wrong here when surrounding lines?
        # TODO: Figure out and fix
        self.replace(mobject, dim_to_match, stretch)

        self.width = np.sqrt(mobject.width ** 2 + mobject.height ** 2)
        return self.scale(buffer_factor)

    def point_at_angle(self, angle):
        """Returns the position of a point on the circle.

        Parameters
        ----------
        angle : class: `float`
            The angle of the point along the circle in radians.

        Examples
        --------

        .. manim:: PointAtAngleExample
            :save_last_frame:

            class PointAtAngleExample(Scene):
                def construct(self):
                    circle = Circle(radius=2.0)
                    p1 = circle.point_at_angle(PI/2)
                    p2 = circle.point_at_angle(270*DEGREES)

                    s1 = Square(side_length=0.25).move_to(p1)
                    s2 = Square(side_length=0.25).move_to(p2)
                    self.add(circle, s1, s2)

        Returns
        -------
        :class:`numpy.ndarray`
            The location of the point along the circle's circumference.
        """

        start_angle = angle_of_vector(self.points[0] - self.get_center())
        return self.point_from_proportion((angle - start_angle) / TAU)

    @staticmethod
    def from_three_points(
        p1: Sequence[float], p2: Sequence[float], p3: Sequence[float], **kwargs
    ):
        """Returns a circle passing through the specified
        three points.

        Example
        -------

        .. manim:: CircleFromPointsExample
            :save_last_frame:

            class CircleFromPointsExample(Scene):
                def construct(self):
                    circle = Circle.from_three_points(LEFT, LEFT + UP, UP * 2, color=RED)
                    dots = VGroup(
                        Dot(LEFT),
                        Dot(LEFT + UP),
                        Dot(UP * 2),
                    )
                    self.add(NumberPlane(), circle, dots)
        """
        center = line_intersection(
            perpendicular_bisector([p1, p2]),
            perpendicular_bisector([p2, p3]),
        )
        radius = np.linalg.norm(p1 - center)
        return Circle(radius=radius, **kwargs).shift(center)


class Dot(Circle):
    """A circle with a very small radius.

    Parameters
    ----------
    point : Union[:class:`list`, :class:`numpy.ndarray`], optional
        The location of the dot.
    radius : Optional[:class:`float`]
        The radius of the dot.
    stroke_width : :class:`float`, optional
        The thickness of the outline of the dot.
    fill_opacity : :class:`float`, optional
        The opacity of the dot's fill_colour
    color : :class:`~.Colors`, optional
        The color of the dot.
    kwargs : Any
        Additional arguments to be passed to :class:`Circle`

    Examples
    --------

    .. manim:: DotExample
        :save_last_frame:

        class DotExample(Scene):
            def construct(self):
                dot1 = Dot(point=LEFT, radius=0.08)
                dot2 = Dot(point=ORIGIN)
                dot3 = Dot(point=RIGHT)
                self.add(dot1,dot2,dot3)
    """

    def __init__(
        self,
        point=ORIGIN,
        radius: float = DEFAULT_DOT_RADIUS,
        stroke_width=0,
        fill_opacity=1.0,
        color=WHITE,
        **kwargs,
    ):
        super().__init__(
            arc_center=point,
            radius=radius,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            color=color,
            **kwargs,
        )


class AnnotationDot(Dot):
    """
    A dot with bigger radius and bold stroke to annotate scenes.
    """

    def __init__(
        self,
        radius: float = DEFAULT_DOT_RADIUS * 1.3,
        stroke_width=5,
        stroke_color=WHITE,
        fill_color=BLUE,
        **kwargs,
    ):
        super().__init__(
            radius=radius,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            fill_color=fill_color,
            **kwargs,
        )


class LabeledDot(Dot):
    """A :class:`Dot` containing a label in its center.

    Parameters
    ----------
    label : Union[:class:`str`, :class:`~.SingleStringMathTex`, :class:`~.Text`, :class:`~.Tex`]
        The label of the :class:`Dot`. This is rendered as :class:`~.MathTex`
        by default (i.e., when passing a :class:`str`), but other classes
        representing rendered strings like :class:`~.Text` or :class:`~.Tex`
        can be passed as well.

    radius : :class:`float`
        The radius of the :class:`Dot`. If ``None`` (the default), the radius
        is calculated based on the size of the ``label``.

    Examples
    --------

    .. manim:: SeveralLabeledDots
        :save_last_frame:

        class SeveralLabeledDots(Scene):
            def construct(self):
                sq = Square(fill_color=RED, fill_opacity=1)
                self.add(sq)
                dot1 = LabeledDot(Tex("42", color=RED))
                dot2 = LabeledDot(MathTex("a", color=GREEN))
                dot3 = LabeledDot(Text("ii", color=BLUE))
                dot4 = LabeledDot("3")
                dot1.next_to(sq, UL)
                dot2.next_to(sq, UR)
                dot3.next_to(sq, DL)
                dot4.next_to(sq, DR)
                self.add(dot1, dot2, dot3, dot4)
    """

    def __init__(self, label, radius=None, **kwargs) -> None:
        if isinstance(label, str):
            from manim import MathTex

            rendered_label = MathTex(label, color=BLACK)
        else:
            rendered_label = label

        if radius is None:
            radius = 0.1 + max(rendered_label.width, rendered_label.height) / 2
        super().__init__(radius=radius, **kwargs)
        rendered_label.move_to(self.get_center())
        self.add(rendered_label)


class Ellipse(Circle):
    """A circular shape; oval, circle.

    Parameters
    ----------
    width : :class:`float`, optional
       The horizontal width of the ellipse.
    height : :class:`float`, optional
       The vertical height of the ellipse.
    kwargs : Any
       Additional arguments to be passed to :class:`Circle`

    Examples
    --------

    .. manim:: EllipseExample
        :save_last_frame:

        class EllipseExample(Scene):
            def construct(self):
                ellipse_1 = Ellipse(width=2.0, height=4.0, color=BLUE_B)
                ellipse_2 = Ellipse(width=4.0, height=1.0, color=BLUE_D)
                ellipse_group = Group(ellipse_1,ellipse_2).arrange(buff=1)
                self.add(ellipse_group)
    """

    def __init__(self, width=2, height=1, **kwargs):
        super().__init__(**kwargs)
        self.stretch_to_fit_width(width)
        self.stretch_to_fit_height(height)


class AnnularSector(Arc):
    """

    Parameters
    ----------
    inner_radius
       The inside radius of the Annular Sector.
    outer_radius
       The outside radius of the Annular Sector.
    angle
       The clockwise angle of the Annular Sector.
    start_angle
       The starting clockwise angle of the Annular Sector.
    fill_opacity
       The opacity of the color filled in the Annular Sector.
    stroke_width
       The stroke width of the Annular Sector.
    color
       The color filled into the Annular Sector.

    Examples
    --------
    .. manim:: AnnularSectorExample
        :save_last_frame:

        class AnnularSectorExample(Scene):
            def construct(self):
                # Changes background color to clearly visualize changes in fill_opacity.
                self.camera.background_color = WHITE

                # The default parameter start_angle is 0, so the AnnularSector starts from the +x-axis.
                s1 = AnnularSector(color=YELLOW).move_to(2 * UL)

                # Different inner_radius and outer_radius than the default.
                s2 = AnnularSector(inner_radius=1.5, outer_radius=2, angle=45 * DEGREES, color=RED).move_to(2 * UR)

                # fill_opacity is typically a number > 0 and <= 1. If fill_opacity=0, the AnnularSector is transparent.
                s3 = AnnularSector(inner_radius=1, outer_radius=1.5, angle=PI, fill_opacity=0.25, color=BLUE).move_to(2 * DL)

                # With a negative value for the angle, the AnnularSector is drawn clockwise from the start value.
                s4 = AnnularSector(inner_radius=1, outer_radius=1.5, angle=-3 * PI / 2, color=GREEN).move_to(2 * DR)

                self.add(s1, s2, s3, s4)


    """

    def __init__(
        self,
        inner_radius=1,
        outer_radius=2,
        angle=TAU / 4,
        start_angle=0,
        fill_opacity=1,
        stroke_width=0,
        color=WHITE,
        **kwargs,
    ):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        super().__init__(
            start_angle=start_angle,
            angle=angle,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            color=color,
            **kwargs,
        )

    def generate_points(self):
        inner_arc, outer_arc = (
            Arc(
                start_angle=self.start_angle,
                angle=self.angle,
                radius=radius,
                arc_center=self.arc_center,
            )
            for radius in (self.inner_radius, self.outer_radius)
        )
        outer_arc.reverse_points()
        self.append_points(inner_arc.points)
        self.add_line_to(outer_arc.points[0])
        self.append_points(outer_arc.points)
        self.add_line_to(inner_arc.points[0])

    init_points = generate_points


class Sector(AnnularSector):
    """

    Examples
    --------

    .. manim:: ExampleSector
        :save_last_frame:

        class ExampleSector(Scene):
            def construct(self):
                sector = Sector(outer_radius=2, inner_radius=1)
                sector2 = Sector(outer_radius=2.5, inner_radius=0.8).move_to([-3, 0, 0])
                sector.set_color(RED)
                sector2.set_color(PINK)
                self.add(sector, sector2)
    """

    def __init__(self, outer_radius=1, inner_radius=0, **kwargs):
        super().__init__(inner_radius=inner_radius, outer_radius=outer_radius, **kwargs)


class Annulus(Circle):
    """Region between two concentric :class:`Circles <.Circle>`.

    Parameters
    ----------
    inner_radius
        The radius of the inner :class:`Circle`.
    outer_radius
        The radius of the outer :class:`Circle`.
    kwargs : Any
        Additional arguments to be passed to :class:`Annulus`

    Examples
    --------

    .. manim:: AnnulusExample
        :save_last_frame:

        class AnnulusExample(Scene):
            def construct(self):
                annulus_1 = Annulus(inner_radius=0.5, outer_radius=1).shift(UP)
                annulus_2 = Annulus(inner_radius=0.3, outer_radius=0.6, color=RED).next_to(annulus_1, DOWN)
                self.add(annulus_1, annulus_2)
    """

    def __init__(
        self,
        inner_radius: Optional[float] = 1,
        outer_radius: Optional[float] = 2,
        fill_opacity=1,
        stroke_width=0,
        color=WHITE,
        mark_paths_closed=False,
        **kwargs,
    ):
        self.mark_paths_closed = mark_paths_closed  # is this even used?
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        super().__init__(
            fill_opacity=fill_opacity, stroke_width=stroke_width, color=color, **kwargs
        )

    def generate_points(self):
        self.radius = self.outer_radius
        outer_circle = Circle(radius=self.outer_radius)
        inner_circle = Circle(radius=self.inner_radius)
        inner_circle.reverse_points()
        self.append_points(outer_circle.points)
        self.append_points(inner_circle.points)
        self.shift(self.arc_center)

    init_points = generate_points


class Line(TipableVMobject):
    def __init__(self, start=LEFT, end=RIGHT, buff=0, path_arc=None, **kwargs):
        self.dim = 3
        self.buff = buff
        self.path_arc = path_arc
        self.set_start_and_end_attrs(start, end)
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

        self.account_for_buff(buff)

    init_points = generate_points

    def set_path_arc(self, new_value):
        self.path_arc = new_value
        self.init_points()

    def account_for_buff(self, buff):
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

    def set_start_and_end_attrs(self, start, end):
        # If either start or end are Mobjects, this
        # gives their centers
        rough_start = self.pointify(start)
        rough_end = self.pointify(end)
        vect = normalize(rough_end - rough_start)
        # Now that we know the direction between them,
        # we can find the appropriate boundary point from
        # start and end, if they're mobjects
        self.start = self.pointify(start, vect)
        self.end = self.pointify(end, -vect)

    def pointify(self, mob_or_point, direction=None):
        if isinstance(mob_or_point, (Mobject, OpenGLMobject)):
            mob = mob_or_point
            if direction is None:
                return mob.get_center()
            else:
                return mob.get_boundary_point(direction)
        return np.array(mob_or_point)

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
    args : Any
        Arguments to be passed to :class:`Line`
    dash_length : :class:`float`, optional
        The length of each individual dash of the line.
    dashed_ratio : :class:`float`, optional
        The ratio of dash space to empty space. Range of 0-1.
    kwargs : Any
        Additional arguments to be passed to :class:`Line`

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

    See Also
    --------
    :class:`~.DashedVMobject`
    """

    @deprecated_params(
        params="positive_space_ratio dash_spacing",
        since="v0.9.0",
        message="Use dashed_ratio instead of positive_space_ratio.",
        redirections=[("positive_space_ratio", "dashed_ratio")],
    )
    def __init__(
        self,
        *args,
        dash_length=DEFAULT_DASH_LENGTH,
        dashed_ratio=0.5,
        **kwargs,
    ):
        self.dash_spacing = kwargs.pop(
            "dash_spacing",
            None,
        )  # Unused param, remove with deprecation warning
        self.dash_length = dash_length
        self.dashed_ratio = dashed_ratio
        super().__init__(*args, **kwargs)
        dashes = DashedVMobject(
            self,
            num_dashes=self.calculate_num_dashes(),
            dashed_ratio=dashed_ratio,
        )
        self.clear_points()
        self.add(*dashes)

    def calculate_num_dashes(self) -> int:
        """Returns the number of dashes in the dashed line.

        Examples
        --------
        ::

            >>> DashedLine().calculate_num_dashes()
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
    vmob : :class:`~.VMobject`
        The VMobject on which the tangent line is drawn.
    alpha : :class:`float`
        How far along the shape that the line will be constructed. range: 0-1.
    length : :class:`float`, optional
        Length of the tangent line.
    d_alpha: :class:`float`, optional
        The ``dx`` value
    kwargs : Any
        Additional arguments to be passed to :class:`Line`

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

    See Also
    --------
    :meth:`~.VMobject.point_from_proportion`
    """

    def __init__(self, vmob, alpha, length=1, d_alpha=1e-6, **kwargs):
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
    width : :class:`float`, optional
        The length of the elbow's sides.
    angle : :class:`float`, optional
        The rotation of the elbow.
    kwargs : Any
        Additional arguments to be passed to :class:`~.VMobject`

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

    See Also
    --------
    :class:`RightAngle`
    """

    def __init__(self, width=0.2, angle=0, **kwargs):
        self.angle = angle
        super().__init__(**kwargs)
        self.set_points_as_corners([UP, UP + RIGHT, RIGHT])
        self.scale_to_fit_width(width, about_point=ORIGIN)
        self.rotate(self.angle, about_point=ORIGIN)


class Arrow(Line):
    """An arrow.

    Parameters
    ----------
    args : Any
        Arguments to be passed to :class:`Line`.
    stroke_width : :class:`float`, optional
        The thickness of the arrow. Influenced by :attr:`max_stroke_width_to_length_ratio`.
    buff : :class:`float`, optional
        The distance of the arrow from its start and end points.
    max_tip_length_to_length_ratio : :class:`float`, optional
        :attr:`tip_length` scales with the length of the arrow. Increasing this ratio raises the max value of :attr:`tip_length`.
    max_stroke_width_to_length_ratio : :class:`float`, optional
        :attr:`stroke_width` scales with the length of the arrow. Increasing this ratio ratios the max value of :attr:`stroke_width`.
    kwargs : Any
        Additional arguments to be passed to :class:`Line`.

    Examples
    --------

    .. manim:: ArrowExample
        :save_last_frame:

        from manim.mobject.geometry import ArrowSquareTip
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


    See Also
    --------
    :class:`ArrowTip`
    :class:`CurvedArrow`
    """

    def __init__(
        self,
        *args,
        stroke_width=6,
        buff=MED_SMALL_BUFF,
        max_tip_length_to_length_ratio=0.25,
        max_stroke_width_to_length_ratio=5,
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
        self.set_stroke_width_from_length()

    def scale(self, factor, scale_tips=False, **kwargs):
        r"""Scale an arrow, but keep stroke width and arrow tip size fixed.

        See Also
        --------
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
            self.set_stroke_width_from_length()
            return self

        has_tip = self.has_tip()
        has_start_tip = self.has_start_tip()
        if has_tip or has_start_tip:
            old_tips = self.pop_tips()

        super().scale(factor, **kwargs)
        self.set_stroke_width_from_length()

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

    def set_stroke_width_from_length(self):
        """Used internally. Sets stroke width based on length."""
        max_ratio = self.max_stroke_width_to_length_ratio
        if config.renderer == "opengl":
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
        self,
        integer_labels: bool = True,
        n_dim: int = 2,
        color: str = WHITE,
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


class DoubleArrow(Arrow):
    """An arrow with tips on both ends.

    Parameters
    ----------
    args : Any
        Arguments to be passed to :class:`Arrow`
    kwargs : Any
        Additional arguments to be passed to :class:`Arrow`

    Examples
    --------

    .. manim:: DoubleArrowExample
        :save_last_frame:

        from manim.mobject.geometry import ArrowCircleFilledTip
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

    See Also
    --------
    :class:`ArrowTip`
    :class:`CurvedDoubleArrow`
    """

    def __init__(self, *args, **kwargs):
        if "tip_shape_end" in kwargs:
            kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
        tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
        super().__init__(*args, **kwargs)
        self.add_tip(at_start=True, tip_shape=tip_shape_start)


class CubicBezier(VMobject, metaclass=ConvertToOpenGL):
    """
    Example
    -------

    .. manim:: BezierSplineExample
        :save_last_frame:

        class BezierSplineExample(Scene):
            def construct(self):
                p1 = np.array([-3, 1, 0])
                p1b = p1 + [1, 0, 0]
                d1 = Dot(point=p1).set_color(BLUE)
                l1 = Line(p1, p1b)
                p2 = np.array([3, -1, 0])
                p2b = p2 - [1, 0, 0]
                d2 = Dot(point=p2).set_color(RED)
                l2 = Line(p2, p2b)
                bezier = CubicBezier(p1b, p1b + 3 * RIGHT, p2b - 3 * RIGHT, p2b)
                self.add(l1, d1, l2, d2, bezier)

    """

    def __init__(self, start_anchor, start_handle, end_handle, end_anchor, **kwargs):
        super().__init__(**kwargs)
        self.add_cubic_bezier_curve(start_anchor, start_handle, end_handle, end_anchor)


class Polygram(VMobject, metaclass=ConvertToOpenGL):
    """A generalized :class:`Polygon`, allowing for disconnected sets of edges.

    Parameters
    ----------
    vertex_groups
        The groups of vertices making up the :class:`Polygram`.

        The first vertex in each group is repeated to close the shape.
        Each point must be 3-dimensional: ``[x,y,z]``
    color
        The color of the :class:`Polygram`.
    kwargs
        Forwarded to the parent constructor.

    Examples
    --------
    .. manim:: PolygramExample

        import numpy as np

        class PolygramExample(Scene):
            def construct(self):
                hexagram = Polygram(
                    [[0, 2, 0], [-np.sqrt(3), -1, 0], [np.sqrt(3), -1, 0]],
                    [[-np.sqrt(3), 1, 0], [0, -2, 0], [np.sqrt(3), 1, 0]],
                )
                self.add(hexagram)

                dot = Dot()
                self.play(MoveAlongPath(dot, hexagram), run_time=5, rate_func=linear)
                self.remove(dot)
                self.wait()


    """

    def __init__(self, *vertex_groups: Iterable[Sequence[float]], color=BLUE, **kwargs):
        super().__init__(color=color, **kwargs)

        for vertices in vertex_groups:
            first_vertex, *vertices = vertices
            first_vertex = np.array(first_vertex)

            self.start_new_path(first_vertex)
            self.add_points_as_corners(
                [*(np.array(vertex) for vertex in vertices), first_vertex],
            )

    def get_vertices(self) -> np.ndarray:
        """Gets the vertices of the :class:`Polygram`.

        Returns
        -------
        :class:`numpy.ndarray`
            The vertices of the :class:`Polygram`.

        Examples
        --------
        ::

            >>> sq = Square()
            >>> sq.get_vertices()
            array([[ 1.,  1.,  0.],
                   [-1.,  1.,  0.],
                   [-1., -1.,  0.],
                   [ 1., -1.,  0.]])
        """

        return self.get_start_anchors()

    def get_vertex_groups(self) -> np.ndarray:
        """Gets the vertex groups of the :class:`Polygram`.

        Returns
        -------
        :class:`numpy.ndarray`
            The vertex groups of the :class:`Polygram`.

        Examples
        --------
        ::

            >>> poly = Polygram([ORIGIN, RIGHT, UP], [LEFT, LEFT + UP, 2 * LEFT])
            >>> poly.get_vertex_groups()
            array([[[ 0.,  0.,  0.],
                    [ 1.,  0.,  0.],
                    [ 0.,  1.,  0.]],
            <BLANKLINE>
                   [[-1.,  0.,  0.],
                    [-1.,  1.,  0.],
                    [-2.,  0.,  0.]]])
        """

        vertex_groups = []

        group = []
        for start, end in zip(self.get_start_anchors(), self.get_end_anchors()):
            group.append(start)

            if self.consider_points_equals(end, group[0]):
                vertex_groups.append(group)
                group = []

        return np.array(vertex_groups)

    def round_corners(self, radius: float = 0.5):
        """Rounds off the corners of the :class:`Polygram`.

        Parameters
        ----------
        radius
            The curvature of the corners of the :class:`Polygram`.

        Examples
        --------

        .. manim:: PolygramRoundCorners
            :save_last_frame:

            class PolygramRoundCorners(Scene):
                def construct(self):
                    star = Star(outer_radius=2)

                    shapes = VGroup(star)
                    shapes.add(star.copy().round_corners(radius=0.1))
                    shapes.add(star.copy().round_corners(radius=0.25))

                    shapes.arrange(RIGHT)
                    self.add(shapes)

        See Also
        --------
        :class:`RoundedRectangle`
        """

        if radius == 0:
            return self

        new_points = []

        for vertices in self.get_vertex_groups():
            arcs = []
            for v1, v2, v3 in adjacent_n_tuples(vertices, 3):
                vect1 = v2 - v1
                vect2 = v3 - v2
                unit_vect1 = normalize(vect1)
                unit_vect2 = normalize(vect2)

                angle = angle_between_vectors(vect1, vect2)
                # Negative radius gives concave curves
                angle *= np.sign(radius)

                # Distance between vertex and start of the arc
                cut_off_length = radius * np.tan(angle / 2)

                # Determines counterclockwise vs. clockwise
                sign = np.sign(np.cross(vect1, vect2)[2])

                arc = ArcBetweenPoints(
                    v2 - unit_vect1 * cut_off_length,
                    v2 + unit_vect2 * cut_off_length,
                    angle=sign * angle,
                )
                arcs.append(arc)

            # To ensure that we loop through starting with last
            arcs = [arcs[-1], *arcs[:-1]]
            for arc1, arc2 in adjacent_pairs(arcs):
                new_points.extend(arc1.points)

                line = Line(arc1.get_end(), arc2.get_start())

                # Make sure anchors are evenly distributed
                len_ratio = line.get_length() / arc1.get_arc_length()

                line.insert_n_curves(int(arc1.get_num_curves() * len_ratio))

                new_points.extend(line.points)

        self.set_points(new_points)

        return self


class Polygon(Polygram):
    """A shape consisting of one closed loop of vertices.

    Parameters
    ----------
    vertices
        The vertices of the :class:`Polygon`.
    kwargs
        Forwarded to the parent constructor.

    Examples
    --------

    .. manim:: PolygonExample
        :save_last_frame:

        class PolygonExample(Scene):
            def construct(self):
                isosceles = Polygon([-5, 1.5, 0], [-2, 1.5, 0], [-3.5, -2, 0])
                position_list = [
                    [4, 1, 0],  # middle right
                    [4, -2.5, 0],  # bottom right
                    [0, -2.5, 0],  # bottom left
                    [0, 3, 0],  # top left
                    [2, 1, 0],  # middle
                    [4, 3, 0],  # top right
                ]
                square_and_triangles = Polygon(*position_list, color=PURPLE_B)
                self.add(isosceles, square_and_triangles)
    """

    def __init__(self, *vertices: Sequence[float], **kwargs):
        super().__init__(vertices, **kwargs)


class RegularPolygram(Polygram):
    """A :class:`Polygram` with regularly spaced vertices.

    Parameters
    ----------
    num_vertices
        The number of vertices.
    density
        The density of the :class:`RegularPolygram`.

        Can be thought of as how many vertices to hop
        to draw a line between them. Every ``density``-th
        vertex is connected.
    radius
        The radius of the circle that the vertices are placed on.
    start_angle
        The angle the vertices start at; the rotation of
        the :class:`RegularPolygram`.
    kwargs
        Forwarded to the parent constructor.

    Examples
    --------
    .. manim:: RegularPolygramExample
        :save_last_frame:

        class RegularPolygramExample(Scene):
            def construct(self):
                pentagram = RegularPolygram(5, radius=2)
                self.add(pentagram)
    """

    def __init__(
        self,
        num_vertices: int,
        *,
        density: int = 2,
        radius: float = 1,
        start_angle: Optional[float] = None,
        **kwargs,
    ):
        # Regular polygrams can be expressed by the number of their vertices
        # and their density. This relation can be expressed as its Schläfli
        # symbol: {num_vertices/density}.
        #
        # For instance, a pentagon can be expressed as {5/1} or just {5}.
        # A pentagram, however, can be expressed as {5/2}.
        # A hexagram *would* be expressed as {6/2}, except that 6 and 2
        # are not coprime, and it can be simplified to 2{3}, which corresponds
        # to the fact that a hexagram is actually made up of 2 triangles.
        #
        # See https://en.wikipedia.org/wiki/Polygram_(geometry)#Generalized_regular_polygons
        # for more information.

        num_gons = np.gcd(num_vertices, density)
        num_vertices //= num_gons
        density //= num_gons

        # Utility function for generating the individual
        # polygon vertices.
        def gen_polygon_vertices(start_angle):
            reg_vertices, start_angle = regular_vertices(
                num_vertices,
                radius=radius,
                start_angle=start_angle,
            )

            vertices = []
            i = 0
            while True:
                vertices.append(reg_vertices[i])

                i += density
                i %= num_vertices
                if i == 0:
                    break

            return vertices, start_angle

        first_group, self.start_angle = gen_polygon_vertices(start_angle)
        vertex_groups = [first_group]

        for i in range(1, num_gons):
            start_angle = self.start_angle + (i / num_gons) * TAU / num_vertices
            group, _ = gen_polygon_vertices(start_angle)

            vertex_groups.append(group)

        super().__init__(*vertex_groups, **kwargs)


class RegularPolygon(RegularPolygram):
    """An n-sided regular :class:`Polygon`.

    Parameters
    ----------
    n
        The number of sides of the :class:`RegularPolygon`.
    kwargs
        Forwarded to the parent constructor.

    Examples
    --------

    .. manim:: RegularPolygonExample
        :save_last_frame:

        class RegularPolygonExample(Scene):
            def construct(self):
                poly_1 = RegularPolygon(n=6)
                poly_2 = RegularPolygon(n=6, start_angle=30*DEGREES, color=GREEN)
                poly_3 = RegularPolygon(n=10, color=RED)

                poly_group = Group(poly_1, poly_2, poly_3).scale(1.5).arrange(buff=1)
                self.add(poly_group)
    """

    def __init__(self, n: int = 6, **kwargs):
        super().__init__(n, density=1, **kwargs)


class Star(Polygon):
    """A regular polygram without the intersecting lines.

    Parameters
    ----------
    n
        How many points on the :class:`Star`.
    outer_radius
        The radius of the circle that the outer vertices are placed on.
    inner_radius
        The radius of the circle that the inner vertices are placed on.

        If unspecified, the inner radius will be
        calculated such that the edges of the :class:`Star`
        perfectly follow the edges of its :class:`RegularPolygram`
        counterpart.
    density
        The density of the :class:`Star`. Only used if
        ``inner_radius`` is unspecified.

        See :class:`RegularPolygram` for more information.
    start_angle
        The angle the vertices start at; the rotation of
        the :class:`Star`.
    kwargs
        Forwardeds to the parent constructor.

    Raises
    ------
    :exc:`ValueError`
        If ``inner_radius`` is unspecified and ``density``
        is not in the range ``[1, n/2)``.

    Examples
    --------
    .. manim:: StarExample
        :save_as_gif:

        class StarExample(Scene):
            def construct(self):
                pentagram = RegularPolygram(5, radius=2)
                star = Star(outer_radius=2, color=RED)

                self.add(pentagram)
                self.play(Create(star), run_time=3)
                self.play(FadeOut(star), run_time=2)

    .. manim:: DifferentDensitiesExample
        :save_last_frame:

        class DifferentDensitiesExample(Scene):
            def construct(self):
                density_2 = Star(7, outer_radius=2, density=2, color=RED)
                density_3 = Star(7, outer_radius=2, density=3, color=PURPLE)

                self.add(VGroup(density_2, density_3).arrange(RIGHT))

    """

    def __init__(
        self,
        n: int = 5,
        *,
        outer_radius: float = 1,
        inner_radius: Optional[float] = None,
        density: int = 2,
        start_angle: Optional[float] = TAU / 4,
        **kwargs,
    ):
        inner_angle = TAU / (2 * n)

        if inner_radius is None:
            # See https://math.stackexchange.com/a/2136292 for an
            # overview of how to calculate the inner radius of a
            # perfect star.

            if density <= 0 or density >= n / 2:
                raise ValueError(
                    f"Incompatible density {density} for number of points {n}",
                )

            outer_angle = TAU * density / n
            inverse_x = 1 - np.tan(inner_angle) * (
                (np.cos(outer_angle) - 1) / np.sin(outer_angle)
            )

            inner_radius = outer_radius / (np.cos(inner_angle) * inverse_x)

        outer_vertices, self.start_angle = regular_vertices(
            n,
            radius=outer_radius,
            start_angle=start_angle,
        )
        inner_vertices, _ = regular_vertices(
            n,
            radius=inner_radius,
            start_angle=self.start_angle + inner_angle,
        )

        vertices = []
        for pair in zip(outer_vertices, inner_vertices):
            vertices.extend(pair)

        super().__init__(*vertices, **kwargs)


class ArcPolygon(VMobject, metaclass=ConvertToOpenGL):
    """A generalized polygon allowing for points to be connected with arcs.

    This version tries to stick close to the way :class:`Polygon` is used. Points
    can be passed to it directly which are used to generate the according arcs
    (using :class:`ArcBetweenPoints`). An angle or radius can be passed to it to
    use across all arcs, but to configure arcs individually an ``arc_config`` list
    has to be passed with the syntax explained below.

    .. tip::

        Two instances of :class:`ArcPolygon` can be transformed properly into one
        another as well. Be advised that any arc initialized with ``angle=0``
        will actually be a straight line, so if a straight section should seamlessly
        transform into an arced section or vice versa, initialize the straight section
        with a negligible angle instead (such as ``angle=0.0001``).

    There is an alternative version (:class:`ArcPolygonFromArcs`) that is instantiated
    with pre-defined arcs.

    See Also
    --------
    :class:`ArcPolygonFromArcs`

    Parameters
    ----------
    vertices : Union[:class:`list`, :class:`np.array`]
        A list of vertices, start and end points for the arc segments.
    angle : :class:`float`
        The angle used for constructing the arcs. If no other parameters
        are set, this angle is used to construct all arcs.
    radius : Optional[:class:`float`]
        The circle radius used to construct the arcs. If specified,
        overrides the specified ``angle``.
    arc_config : Optional[Union[List[:class:`dict`]], :class:`dict`]
        When passing a ``dict``, its content will be passed as keyword
        arguments to :class:`~.ArcBetweenPoints`. Otherwise, a list
        of dictionaries containing values that are passed as keyword
        arguments for every individual arc can be passed.
    kwargs
        Further keyword arguments that are passed to the constructor of
        :class:`~.VMobject`.

    Attributes
    ----------
    arcs : :class:`list`
        The arcs created from the input parameters::

            >>> from manim import ArcPolygon
            >>> ap = ArcPolygon([0, 0, 0], [2, 0, 0], [0, 2, 0])
            >>> ap.arcs
            [ArcBetweenPoints, ArcBetweenPoints, ArcBetweenPoints]

    Examples
    --------

    .. manim:: SeveralArcPolygons

        class SeveralArcPolygons(Scene):
            def construct(self):
                a = [0, 0, 0]
                b = [2, 0, 0]
                c = [0, 2, 0]
                ap1 = ArcPolygon(a, b, c, radius=2)
                ap2 = ArcPolygon(a, b, c, angle=45*DEGREES)
                ap3 = ArcPolygon(a, b, c, arc_config={'radius': 1.7, 'color': RED})
                ap4 = ArcPolygon(a, b, c, color=RED, fill_opacity=1,
                                            arc_config=[{'radius': 1.7, 'color': RED},
                                            {'angle': 20*DEGREES, 'color': BLUE},
                                            {'radius': 1}])
                ap_group = VGroup(ap1, ap2, ap3, ap4).arrange()
                self.play(*[Create(ap) for ap in [ap1, ap2, ap3, ap4]])
                self.wait()

    For further examples see :class:`ArcPolygonFromArcs`.
    """

    def __init__(self, *vertices, angle=PI / 4, radius=None, arc_config=None, **kwargs):
        n = len(vertices)
        point_pairs = [(vertices[k], vertices[(k + 1) % n]) for k in range(n)]

        if not arc_config:
            if radius:
                all_arc_configs = itertools.repeat({"radius": radius}, len(point_pairs))
            else:
                all_arc_configs = itertools.repeat({"angle": angle}, len(point_pairs))
        elif isinstance(arc_config, dict):
            all_arc_configs = itertools.repeat(arc_config, len(point_pairs))
        else:
            assert len(arc_config) == n
            all_arc_configs = arc_config

        arcs = [
            ArcBetweenPoints(*pair, **conf)
            for (pair, conf) in zip(point_pairs, all_arc_configs)
        ]

        super().__init__(**kwargs)
        # Adding the arcs like this makes ArcPolygon double as a VGroup.
        # Also makes changes to the ArcPolygon, such as scaling, affect
        # the arcs, so that their new values are usable.
        self.add(*arcs)
        for arc in arcs:
            self.append_points(arc.points)

        # This enables the use of ArcPolygon.arcs as a convenience
        # because ArcPolygon[0] returns itself, not the first Arc.
        self.arcs = arcs


class ArcPolygonFromArcs(VMobject, metaclass=ConvertToOpenGL):
    """A generalized polygon allowing for points to be connected with arcs.

    This version takes in pre-defined arcs to generate the arcpolygon and introduces
    little new syntax. However unlike :class:`Polygon` it can't be created with points
    directly.

    For proper appearance the passed arcs should connect seamlessly:
    ``[a,b][b,c][c,a]``

    If there are any gaps between the arcs, those will be filled in
    with straight lines, which can be used deliberately for any straight
    sections. Arcs can also be passed as straight lines such as an arc
    initialized with ``angle=0``.

    .. tip::

        Two instances of :class:`ArcPolygon` can be transformed properly into
        one another as well. Be advised that any arc initialized with ``angle=0``
        will actually be a straight line, so if a straight section should seamlessly
        transform into an arced section or vice versa, initialize the straight
        section with a negligible angle instead (such as ``angle=0.0001``).

    There is an alternative version (:class:`ArcPolygon`) that can be instantiated
    with points.

    See Also
    --------
    :class:`ArcPolygon`

    Parameters
    ----------
    arcs : Union[:class:`Arc`, :class:`ArcBetweenPoints`]
        These are the arcs from which the arcpolygon is assembled.
    kwargs
        Keyword arguments that are passed to the constructor of
        :class:`~.VMobject`. Affects how the ArcPolygon itself is drawn,
        but doesn't affect passed arcs.

    Attributes
    ----------
    arcs : :class:`list`
        The arcs used to initialize the ArcPolygonFromArcs::

            >>> from manim import ArcPolygonFromArcs, Arc, ArcBetweenPoints
            >>> ap = ArcPolygonFromArcs(Arc(), ArcBetweenPoints([1,0,0], [0,1,0]), Arc())
            >>> ap.arcs
            [Arc, ArcBetweenPoints, Arc]

    Examples
    --------
    One example of an arcpolygon is the Reuleaux triangle.
    Instead of 3 straight lines connecting the outer points,
    a Reuleaux triangle has 3 arcs connecting those points,
    making a shape with constant width.

    Passed arcs are stored as submobjects in the arcpolygon.
    This means that the arcs are changed along with the arcpolygon,
    for example when it's shifted, and these arcs can be manipulated
    after the arcpolygon has been initialized.

    Also both the arcs contained in an :class:`~.ArcPolygonFromArcs`, as well as the
    arcpolygon itself are drawn, which affects draw time in :class:`~.Create`
    for example. In most cases the arcs themselves don't
    need to be drawn, in which case they can be passed as invisible.

    .. manim:: ArcPolygonExample

        class ArcPolygonExample(Scene):
            def construct(self):
                arc_conf = {"stroke_width": 0}
                poly_conf = {"stroke_width": 10, "stroke_color": BLUE,
                      "fill_opacity": 1, "color": PURPLE}
                a = [-1, 0, 0]
                b = [1, 0, 0]
                c = [0, np.sqrt(3), 0]
                arc0 = ArcBetweenPoints(a, b, radius=2, **arc_conf)
                arc1 = ArcBetweenPoints(b, c, radius=2, **arc_conf)
                arc2 = ArcBetweenPoints(c, a, radius=2, **arc_conf)
                reuleaux_tri = ArcPolygonFromArcs(arc0, arc1, arc2, **poly_conf)
                self.play(FadeIn(reuleaux_tri))
                self.wait(2)

    The arcpolygon itself can also be hidden so that instead only the contained
    arcs are drawn. This can be used to easily debug arcs or to highlight them.

    .. manim:: ArcPolygonExample2

        class ArcPolygonExample2(Scene):
            def construct(self):
                arc_conf = {"stroke_width": 3, "stroke_color": BLUE,
                    "fill_opacity": 0.5, "color": GREEN}
                poly_conf = {"color": None}
                a = [-1, 0, 0]
                b = [1, 0, 0]
                c = [0, np.sqrt(3), 0]
                arc0 = ArcBetweenPoints(a, b, radius=2, **arc_conf)
                arc1 = ArcBetweenPoints(b, c, radius=2, **arc_conf)
                arc2 = ArcBetweenPoints(c, a, radius=2, stroke_color=RED)
                reuleaux_tri = ArcPolygonFromArcs(arc0, arc1, arc2, **poly_conf)
                self.play(FadeIn(reuleaux_tri))
                self.wait(2)

    """

    def __init__(self, *arcs, **kwargs):
        if not all(isinstance(m, (Arc, ArcBetweenPoints)) for m in arcs):
            raise ValueError(
                "All ArcPolygon submobjects must be of type Arc/ArcBetweenPoints",
            )
        super().__init__(**kwargs)
        # Adding the arcs like this makes ArcPolygonFromArcs double as a VGroup.
        # Also makes changes to the ArcPolygonFromArcs, such as scaling, affect
        # the arcs, so that their new values are usable.
        self.add(*arcs)
        # This enables the use of ArcPolygonFromArcs.arcs as a convenience
        # because ArcPolygonFromArcs[0] returns itself, not the first Arc.
        self.arcs = [*arcs]
        for arc1, arc2 in adjacent_pairs(arcs):
            self.append_points(arc1.points)
            line = Line(arc1.get_end(), arc2.get_start())
            len_ratio = line.get_length() / arc1.get_arc_length()
            if math.isnan(len_ratio) or math.isinf(len_ratio):
                continue
            line.insert_n_curves(int(arc1.get_num_curves() * len_ratio))
            self.append_points(line.points)


class Triangle(RegularPolygon):
    """An equilateral triangle.

    Parameters
    ----------
    kwargs : Any
        Additional arguments to be passed to :class:`RegularPolygon`

    Examples
    --------

    .. manim:: TriangleExample
        :save_last_frame:

        class TriangleExample(Scene):
            def construct(self):
                triangle_1 = Triangle()
                triangle_2 = Triangle().scale(2).rotate(60*DEGREES)
                tri_group = Group(triangle_1, triangle_2).arrange(buff=1)
                self.add(tri_group)
    """

    def __init__(self, **kwargs):
        super().__init__(n=3, **kwargs)


class Rectangle(Polygon):
    """A quadrilateral with two sets of parallel sides.

    Parameters
    ----------
    color : :class:`~.Colors`, optional
        The color of the rectangle.
    height : :class:`float`, optional
        The vertical height of the rectangle.
    width : :class:`float`, optional
        The horizontal width of the rectangle.
    grid_xstep : :class:`float`, optional
        Space between vertical grid lines.
    grid_ystep : :class:`float`, optional
        Space between horizontal grid lines.
    mark_paths_closed : :class:`bool`, optional
        No purpose.
    close_new_points : :class:`bool`, optional
        No purpose.
    kwargs : Any
        Additional arguments to be passed to :class:`Polygon`

    Examples
    ----------

    .. manim:: RectangleExample
        :save_last_frame:

        class RectangleExample(Scene):
            def construct(self):
                rect1 = Rectangle(width=4.0, height=2.0, grid_xstep=1.0, grid_ystep=0.5)
                rect2 = Rectangle(width=1.0, height=4.0)

                rects = Group(rect1,rect2).arrange(buff=1)
                self.add(rects)
    """

    def __init__(
        self,
        color: Color = WHITE,
        height: float = 2.0,
        width: float = 4.0,
        grid_xstep: Optional[float] = None,
        grid_ystep: Optional[float] = None,
        mark_paths_closed=True,
        close_new_points=True,
        **kwargs,
    ):
        super().__init__(UR, UL, DL, DR, color=color, **kwargs)
        self.stretch_to_fit_width(width)
        self.stretch_to_fit_height(height)
        v = self.get_vertices()
        if grid_xstep is not None:
            grid_xstep = abs(grid_xstep)
            count = int(width / grid_xstep)
            grid = VGroup(
                *(
                    Line(
                        v[1] + i * grid_xstep * RIGHT,
                        v[1] + i * grid_xstep * RIGHT + height * DOWN,
                        color=color,
                    )
                    for i in range(1, count)
                )
            )
            self.add(grid)
        if grid_ystep is not None:
            grid_ystep = abs(grid_ystep)
            count = int(height / grid_ystep)
            grid = VGroup(
                *(
                    Line(
                        v[1] + i * grid_ystep * DOWN,
                        v[1] + i * grid_ystep * DOWN + width * RIGHT,
                        color=color,
                    )
                    for i in range(1, count)
                )
            )
            self.add(grid)


class Square(Rectangle):
    """A rectangle with equal side lengths.

    Parameters
    ----------
    side_length : :class:`float`, optional
        The length of the sides of the square.
    kwargs : Any
        Additional arguments to be passed to :class:`Square`

    Examples
    --------

    .. manim:: SquareExample
        :save_last_frame:

        class SquareExample(Scene):
            def construct(self):
                square_1 = Square(side_length=2.0).shift(DOWN)
                square_2 = Square(side_length=1.0).next_to(square_1, direction=UP)
                square_3 = Square(side_length=0.5).next_to(square_2, direction=UP)
                self.add(square_1, square_2, square_3)
    """

    def __init__(self, side_length=2.0, **kwargs):
        self.side_length = side_length
        super().__init__(height=side_length, width=side_length, **kwargs)


class RoundedRectangle(Rectangle):
    """A rectangle with rounded corners.

    Parameters
    ----------
    corner_radius : :class:`float`, optional
        The curvature of the corners of the rectangle.
    kwargs : Any
        Additional arguments to be passed to :class:`Rectangle`

    Examples
    --------

    .. manim:: RoundedRectangleExample
        :save_last_frame:

        class RoundedRectangleExample(Scene):
            def construct(self):
                rect_1 = RoundedRectangle(corner_radius=0.5)
                rect_2 = RoundedRectangle(corner_radius=1.5, height=4.0, width=4.0)

                rect_group = Group(rect_1, rect_2).arrange(buff=1)
                self.add(rect_group)
    """

    def __init__(self, corner_radius=0.5, **kwargs):
        self.corner_radius = corner_radius
        super().__init__(**kwargs)
        self.round_corners(self.corner_radius)


class ArrowTip(VMobject, metaclass=ConvertToOpenGL):
    r"""Base class for arrow tips.

    See Also
    --------
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

        >>> class MyCustomArrowTip(ArrowTip, RegularPolygon):
        ...     def __init__(self, length=0.35, **kwargs):
        ...         RegularPolygon.__init__(self, n=5, **kwargs)
        ...         self.width = length
        ...         self.stretch_to_fit_height(length)
        >>> arr = Arrow(np.array([-2, -2, 0]), np.array([2, 2, 0]),
        ...             tip_shape=MyCustomArrowTip)
        >>> isinstance(arr.tip, RegularPolygon)
        True
        >>> from manim import Scene
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

        from manim.mobject.geometry import ArrowTriangleTip, ArrowSquareTip, ArrowSquareFilledTip,\
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
        self.width = length
        self.stretch_to_fit_height(length)


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


class Cutout(VMobject, metaclass=ConvertToOpenGL):
    """A shape with smaller cutouts.

    .. warning::

        Technically, this class behaves similar to a symmetric difference: if
        parts of the ``mobjects`` are not located within the ``main_shape``,
        these parts will be added to the resulting :class:`~.VMobject`.

    Parameters
    ----------
    main_shape : :class:`~.VMobject`
        The primary shape from which cutouts are made.
    mobjects : :class:`~.VMobject`
        The smaller shapes which are to be cut out of the ``main_shape``.
    kwargs
        Further keyword arguments that are passed to the constructor of
        :class:`~.VMobject`.

    Examples
    --------
    .. manim:: CutoutExample

        class CutoutExample(Scene):
            def construct(self):
                s1 = Square().scale(2.5)
                s2 = Triangle().shift(DOWN + RIGHT).scale(0.5)
                s3 = Square().shift(UP + RIGHT).scale(0.5)
                s4 = RegularPolygon(5).shift(DOWN + LEFT).scale(0.5)
                s5 = RegularPolygon(6).shift(UP + LEFT).scale(0.5)
                c = Cutout(s1, s2, s3, s4, s5, fill_opacity=1, color=BLUE, stroke_color=RED)
                self.play(Write(c), run_time=4)
                self.wait()
    """

    def __init__(self, main_shape, *mobjects, **kwargs):
        super().__init__(**kwargs)
        self.append_points(main_shape.points)
        if main_shape.get_direction() == "CW":
            sub_direction = "CCW"
        else:
            sub_direction = "CW"
        for mobject in mobjects:
            self.append_points(mobject.force_direction(sub_direction).points)


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
    quadrant : Sequence[:class:`int`]
        A sequence of two :class:`int` numbers determining which of the 4 quadrants should be used.
        The first value indicates whether to anchor the arc on the first line closer to the end point (1)
        or start point (-1), and the second value functions similarly for the
        end (1) or start (-1) of the second line.
        Possibilities: (1,1), (-1,1), (1,-1), (-1,-1).
    other_angle :
        Toggles between the two possible angles defined by two points and an arc center. If set to
        False (default), the arc will always go counterclockwise from the point on line1 until
        the point on line2 is reached. If set to True, the angle will go clockwise from line1 to line2.
    dot : :class:`bool`
        Allows for a :class:`Dot` in the arc. Mainly used as an convention to indicate a right angle.
        The dot can be customized in the next three parameters.
    dot_radius : :class:`float`
        The radius of the :class:`Dot`. If not specified otherwise, this radius will be 1/10 of the arc radius.
    dot_distance : :class:`float`
        Relative distance from the center to the arc: 0 puts the dot in the center and 1 on the arc itself.
    dot_color : :class:`~.Colors`
        The color of the :class:`Dot`.
    elbow : :class:`bool`
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
        quadrant=(1, 1),
        other_angle: bool = False,
        dot=False,
        dot_radius=None,
        dot_distance=0.55,
        dot_color=WHITE,
        elbow=False,
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


class RightAngle(Angle):
    """An elbow-type mobject representing a right angle between two lines.

    Parameters
    ----------
    line1 : :class:`Line`
        The first line.
    line2 : :class:`Line`
        The second line.
    length : :class:`float`
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

    def __init__(self, line1, line2, length=None, **kwargs):
        super().__init__(line1, line2, radius=length, elbow=True, **kwargs)
