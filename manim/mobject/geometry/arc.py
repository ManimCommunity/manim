r"""Mobjects that are curved.

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

from __future__ import annotations

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
    "CubicBezier",
    "ArcPolygon",
    "ArcPolygonFromArcs",
]

import itertools
import warnings
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from manim.constants import *
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.color import BLACK, BLUE, RED, WHITE, ParsableManimColor
from manim.utils.iterables import adjacent_pairs
from manim.utils.space_ops import (
    angle_of_vector,
    cartesian_to_spherical,
    line_intersection,
    perpendicular_bisector,
    rotate_vector,
)

if TYPE_CHECKING:
    import manim.mobject.geometry.tips as tips
    from manim.mobject.mobject import Mobject
    from manim.mobject.text.tex_mobject import SingleStringMathTex, Tex
    from manim.mobject.text.text_mobject import Text
    from manim.typing import CubicBezierPoints, Point3D, QuadraticBezierPoints, Vector


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
        tip_length: float = DEFAULT_ARROW_TIP_LENGTH,
        normal_vector: Vector = OUT,
        tip_style: dict = {},
        **kwargs,
    ) -> None:
        self.tip_length: float = tip_length
        self.normal_vector: Vector = normal_vector
        self.tip_style: dict = tip_style
        super().__init__(**kwargs)

    # Adding, Creating, Modifying tips

    def add_tip(
        self,
        tip: tips.ArrowTip | None = None,
        tip_shape: type[tips.ArrowTip] | None = None,
        tip_length: float | None = None,
        tip_width: float | None = None,
        at_start: bool = False,
    ) -> Self:
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
        self,
        tip_shape: type[tips.ArrowTip] | None = None,
        tip_length: float = None,
        tip_width: float = None,
        at_start: bool = False,
    ):
        """Stylises the tip, positions it spatially, and returns
        the newly instantiated tip to the caller.
        """
        tip = self.get_unpositioned_tip(tip_shape, tip_length, tip_width)
        self.position_tip(tip, at_start)
        return tip

    def get_unpositioned_tip(
        self,
        tip_shape: type[tips.ArrowTip] | None = None,
        tip_length: float | None = None,
        tip_width: float | None = None,
    ):
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

    def position_tip(self, tip: tips.ArrowTip, at_start: bool = False):
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

    def reset_endpoints_based_on_tip(self, tip: tips.ArrowTip, at_start: bool) -> Self:
        if self.get_length() == 0:
            # Zero length, put_start_and_end_on wouldn't work
            return self

        if at_start:
            self.put_start_and_end_on(tip.base, self.get_end())
        else:
            self.put_start_and_end_on(self.get_start(), tip.base)
        return self

    def asign_tip_attr(self, tip: tips.ArrowTip, at_start: bool) -> Self:
        if at_start:
            self.start_tip = tip
        else:
            self.tip = tip
        return self

    # Checking for tips

    def has_tip(self) -> bool:
        return hasattr(self, "tip") and self.tip in self

    def has_start_tip(self) -> bool:
        return hasattr(self, "start_tip") and self.start_tip in self

    # Getters

    def pop_tips(self) -> VGroup:
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

    def get_tips(self) -> VGroup:
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

    def get_default_tip_length(self) -> float:
        return self.tip_length

    def get_first_handle(self) -> Point3D:
        return self.points[1]

    def get_last_handle(self) -> Point3D:
        return self.points[-2]

    def get_end(self) -> Point3D:
        if self.has_tip():
            return self.tip.get_start()
        else:
            return super().get_end()

    def get_start(self) -> Point3D:
        if self.has_start_tip():
            return self.start_tip.get_start()
        else:
            return super().get_start()

    def get_length(self) -> np.floating:
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
        start_angle: float = 0,
        angle: float = TAU / 4,
        num_components: int = 9,
        arc_center: Point3D = ORIGIN,
        **kwargs,
    ):
        if radius is None:  # apparently None is passed by ArcBetweenPoints
            radius = 1.0
        self.radius = radius
        self.num_components: int = num_components
        self.arc_center: Point3D = arc_center
        self.start_angle: float = start_angle
        self.angle: float = angle
        self._failed_to_get_center: bool = False
        super().__init__(**kwargs)

    def generate_points(self) -> None:
        self._set_pre_positioned_points()
        self.scale(self.radius, about_point=ORIGIN)
        self.shift(self.arc_center)

    # Points are set a bit differently when rendering via OpenGL.
    # TODO: refactor Arc so that only one strategy for setting points
    # has to be used.
    def init_points(self) -> None:
        self.set_points(
            Arc._create_quadratic_bezier_points(
                angle=self.angle,
                start_angle=self.start_angle,
                n_components=self.num_components,
            ),
        )
        self.scale(self.radius, about_point=ORIGIN)
        self.shift(self.arc_center)

    @staticmethod
    def _create_quadratic_bezier_points(
        angle: float, start_angle: float = 0, n_components: int = 8
    ) -> QuadraticBezierPoints:
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

    def _set_pre_positioned_points(self) -> None:
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

    def get_arc_center(self, warning: bool = True) -> Point3D:
        """Looks at the normals to the first two
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

    def move_arc_center_to(self, point: Point3D) -> Self:
        self.shift(point - self.get_arc_center())
        return self

    def stop_angle(self) -> float:
        return angle_of_vector(self.points[-1] - self.get_arc_center()) % TAU


class ArcBetweenPoints(Arc):
    """Inherits from Arc and additionally takes 2 points between which the arc is spanned.

    Example
    -------
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

    def __init__(
        self,
        start: Point3D,
        end: Point3D,
        angle: float = TAU / 4,
        radius: float = None,
        **kwargs,
    ) -> None:
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
            arc_height = radius - np.sqrt(radius**2 - halfdist**2)
            angle = np.arccos((radius - arc_height) / radius) * sign

        super().__init__(radius=radius, angle=angle, **kwargs)
        if angle == 0:
            self.set_points_as_corners([LEFT, RIGHT])
        self.put_start_and_end_on(start, end)

        if radius is None:
            center = self.get_arc_center(warning=False)
            if not self._failed_to_get_center:
                self.radius = np.linalg.norm(np.array(start) - np.array(center))
            else:
                self.radius = np.inf


class CurvedArrow(ArcBetweenPoints):
    def __init__(self, start_point: Point3D, end_point: Point3D, **kwargs) -> None:
        from manim.mobject.geometry.tips import ArrowTriangleFilledTip

        tip_shape = kwargs.pop("tip_shape", ArrowTriangleFilledTip)
        super().__init__(start_point, end_point, **kwargs)
        self.add_tip(tip_shape=tip_shape)


class CurvedDoubleArrow(CurvedArrow):
    def __init__(self, start_point: Point3D, end_point: Point3D, **kwargs) -> None:
        if "tip_shape_end" in kwargs:
            kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
        from manim.mobject.geometry.tips import ArrowTriangleFilledTip

        tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
        super().__init__(start_point, end_point, **kwargs)
        self.add_tip(at_start=True, tip_shape=tip_shape_start)


class Circle(Arc):
    """A circle.

    Parameters
    ----------
    color
        The color of the shape.
    kwargs
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
        radius: float | None = None,
        color: ParsableManimColor = RED,
        **kwargs,
    ) -> None:
        super().__init__(
            radius=radius,
            start_angle=0,
            angle=TAU,
            color=color,
            **kwargs,
        )

    def surround(
        self,
        mobject: Mobject,
        dim_to_match: int = 0,
        stretch: bool = False,
        buffer_factor: float = 1.2,
    ) -> Self:
        """Modifies a circle so that it surrounds a given mobject.

        Parameters
        ----------
        mobject
            The mobject that the circle will be surrounding.
        dim_to_match
        buffer_factor
            Scales the circle with respect to the mobject. A `buffer_factor` < 1 makes the circle smaller than the mobject.
        stretch
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

        self.width = np.sqrt(mobject.width**2 + mobject.height**2)
        return self.scale(buffer_factor)

    def point_at_angle(self, angle: float) -> Point3D:
        """Returns the position of a point on the circle.

        Parameters
        ----------
        angle
            The angle of the point along the circle in radians.

        Returns
        -------
        :class:`numpy.ndarray`
            The location of the point along the circle's circumference.

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

        """

        start_angle = angle_of_vector(self.points[0] - self.get_center())
        proportion = (angle - start_angle) / TAU
        proportion -= np.floor(proportion)
        return self.point_from_proportion(proportion)

    @staticmethod
    def from_three_points(p1: Point3D, p2: Point3D, p3: Point3D, **kwargs) -> Self:
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
    point
        The location of the dot.
    radius
        The radius of the dot.
    stroke_width
        The thickness of the outline of the dot.
    fill_opacity
        The opacity of the dot's fill_colour
    color
        The color of the dot.
    kwargs
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
        point: Point3D = ORIGIN,
        radius: float = DEFAULT_DOT_RADIUS,
        stroke_width: float = 0,
        fill_opacity: float = 1.0,
        color: ParsableManimColor = WHITE,
        **kwargs,
    ) -> None:
        super().__init__(
            arc_center=point,
            radius=radius,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            color=color,
            **kwargs,
        )


class AnnotationDot(Dot):
    """A dot with bigger radius and bold stroke to annotate scenes."""

    def __init__(
        self,
        radius: float = DEFAULT_DOT_RADIUS * 1.3,
        stroke_width: float = 5,
        stroke_color: ParsableManimColor = WHITE,
        fill_color: ParsableManimColor = BLUE,
        **kwargs,
    ) -> None:
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
    label
        The label of the :class:`Dot`. This is rendered as :class:`~.MathTex`
        by default (i.e., when passing a :class:`str`), but other classes
        representing rendered strings like :class:`~.Text` or :class:`~.Tex`
        can be passed as well.
    radius
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

    def __init__(
        self,
        label: str | SingleStringMathTex | Text | Tex,
        radius: float | None = None,
        **kwargs,
    ) -> None:
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
    width
       The horizontal width of the ellipse.
    height
       The vertical height of the ellipse.
    kwargs
       Additional arguments to be passed to :class:`Circle`.

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

    def __init__(self, width: float = 2, height: float = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stretch_to_fit_width(width)
        self.stretch_to_fit_height(height)


class AnnularSector(Arc):
    """A sector of an annulus.


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
        inner_radius: float = 1,
        outer_radius: float = 2,
        angle: float = TAU / 4,
        start_angle: float = 0,
        fill_opacity: float = 1,
        stroke_width: float = 0,
        color: ParsableManimColor = WHITE,
        **kwargs,
    ) -> None:
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

    def generate_points(self) -> None:
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
    """A sector of a circle.

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

    def __init__(
        self, outer_radius: float = 1, inner_radius: float = 0, **kwargs
    ) -> None:
        super().__init__(inner_radius=inner_radius, outer_radius=outer_radius, **kwargs)


class Annulus(Circle):
    """Region between two concentric :class:`Circles <.Circle>`.

    Parameters
    ----------
    inner_radius
        The radius of the inner :class:`Circle`.
    outer_radius
        The radius of the outer :class:`Circle`.
    kwargs
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
        inner_radius: float | None = 1,
        outer_radius: float | None = 2,
        fill_opacity: float = 1,
        stroke_width: float = 0,
        color: ParsableManimColor = WHITE,
        mark_paths_closed: bool = False,
        **kwargs,
    ) -> None:
        self.mark_paths_closed = mark_paths_closed  # is this even used?
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        super().__init__(
            fill_opacity=fill_opacity, stroke_width=stroke_width, color=color, **kwargs
        )

    def generate_points(self) -> None:
        self.radius = self.outer_radius
        outer_circle = Circle(radius=self.outer_radius)
        inner_circle = Circle(radius=self.inner_radius)
        inner_circle.reverse_points()
        self.append_points(outer_circle.points)
        self.append_points(inner_circle.points)
        self.shift(self.arc_center)

    init_points = generate_points


class CubicBezier(VMobject, metaclass=ConvertToOpenGL):
    """A cubic Bézier curve.

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

    def __init__(
        self,
        start_anchor: CubicBezierPoints,
        start_handle: CubicBezierPoints,
        end_handle: CubicBezierPoints,
        end_anchor: CubicBezierPoints,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_cubic_bezier_curve(start_anchor, start_handle, end_handle, end_anchor)


class ArcPolygon(VMobject, metaclass=ConvertToOpenGL):
    """A generalized polygon allowing for points to be connected with arcs.

    This version tries to stick close to the way :class:`Polygon` is used. Points
    can be passed to it directly which are used to generate the according arcs
    (using :class:`ArcBetweenPoints`). An angle or radius can be passed to it to
    use across all arcs, but to configure arcs individually an ``arc_config`` list
    has to be passed with the syntax explained below.

    Parameters
    ----------
    vertices
        A list of vertices, start and end points for the arc segments.
    angle
        The angle used for constructing the arcs. If no other parameters
        are set, this angle is used to construct all arcs.
    radius
        The circle radius used to construct the arcs. If specified,
        overrides the specified ``angle``.
    arc_config
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


    .. tip::

        Two instances of :class:`ArcPolygon` can be transformed properly into one
        another as well. Be advised that any arc initialized with ``angle=0``
        will actually be a straight line, so if a straight section should seamlessly
        transform into an arced section or vice versa, initialize the straight section
        with a negligible angle instead (such as ``angle=0.0001``).

    .. note::
        There is an alternative version (:class:`ArcPolygonFromArcs`) that is instantiated
        with pre-defined arcs.

    See Also
    --------
    :class:`ArcPolygonFromArcs`


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

    def __init__(
        self,
        *vertices: Point3D,
        angle: float = PI / 4,
        radius: float | None = None,
        arc_config: list[dict] | None = None,
        **kwargs,
    ) -> None:
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

    Parameters
    ----------
    arcs
        These are the arcs from which the arcpolygon is assembled.
    kwargs
        Keyword arguments that are passed to the constructor of
        :class:`~.VMobject`. Affects how the ArcPolygon itself is drawn,
        but doesn't affect passed arcs.

    Attributes
    ----------
    arcs
        The arcs used to initialize the ArcPolygonFromArcs::

            >>> from manim import ArcPolygonFromArcs, Arc, ArcBetweenPoints
            >>> ap = ArcPolygonFromArcs(Arc(), ArcBetweenPoints([1,0,0], [0,1,0]), Arc())
            >>> ap.arcs
            [Arc, ArcBetweenPoints, Arc]


    .. tip::

        Two instances of :class:`ArcPolygon` can be transformed properly into
        one another as well. Be advised that any arc initialized with ``angle=0``
        will actually be a straight line, so if a straight section should seamlessly
        transform into an arced section or vice versa, initialize the straight
        section with a negligible angle instead (such as ``angle=0.0001``).

    .. note::
        There is an alternative version (:class:`ArcPolygon`) that can be instantiated
        with points.

    .. seealso::
        :class:`ArcPolygon`

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

    def __init__(self, *arcs: Arc | ArcBetweenPoints, **kwargs) -> None:
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
        from .line import Line

        for arc1, arc2 in adjacent_pairs(arcs):
            self.append_points(arc1.points)
            line = Line(arc1.get_end(), arc2.get_start())
            len_ratio = line.get_length() / arc1.get_arc_length()
            if np.isnan(len_ratio) or np.isinf(len_ratio):
                continue
            line.insert_n_curves(int(arc1.get_num_curves() * len_ratio))
            self.append_points(line.points)
