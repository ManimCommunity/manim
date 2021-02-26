r"""Mobjects that are simple geometric shapes.

Examples
--------

.. manim:: UsefulAnnotations
    :save_last_frame:

    class UsefulAnnotations(Scene):
        def construct(self):
            m0 = SmallDot()
            m1 = AnnotationDot()
            m2 = LabeledDot("ii")
            m3 = LabeledDot(MathTex(r"\alpha").set_color(ORANGE))
            m4 = CurvedArrow(ORIGIN, 2*LEFT)
            m5 = CurvedDoubleArrow(ORIGIN, 2*RIGHT)

            self.add(m0, m1, m2, m3, m4, m5)
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
    "SmallDot",
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
    "Polygon",
    "RegularPolygon",
    "ArcPolygon",
    "ArcPolygonFromArcs",
    "Triangle",
    "ArrowTip",
    "Rectangle",
    "Square",
    "RoundedRectangle",
    "Cutout",
]

import warnings
import numpy as np
import math

from ..constants import *
from ..mobject.mobject import Mobject
from ..mobject.types.vectorized_mobject import VGroup
from ..mobject.types.vectorized_mobject import VMobject
from ..mobject.types.vectorized_mobject import DashedVMobject
from ..utils.iterables import adjacent_n_tuples
from ..utils.iterables import adjacent_pairs
from ..utils.simple_functions import fdiv
from ..utils.space_ops import angle_of_vector
from ..utils.space_ops import angle_between_vectors
from ..utils.space_ops import compass_directions
from ..utils.space_ops import line_intersection
from ..utils.space_ops import get_norm
from ..utils.space_ops import normalize
from ..utils.space_ops import rotate_vector
from ..utils.color import *

DEFAULT_DOT_RADIUS = 0.08
DEFAULT_SMALL_DOT_RADIUS = 0.04
DEFAULT_DASH_LENGTH = 0.05
DEFAULT_ARROW_TIP_LENGTH = 0.35


class TipableVMobject(VMobject):
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
        **kwargs
    ):
        self.tip_length = tip_length
        self.normal_vector = normal_vector
        self.tip_style = tip_style
        VMobject.__init__(self, **kwargs)

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
        tip.rotate(angle_of_vector(handle - anchor) - PI - tip.tip_angle)
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
        result = VGroup()
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
        result = VGroup()
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
            return VMobject.get_end(self)

    def get_start(self):
        if self.has_start_tip():
            return self.start_tip.get_start()
        else:
            return VMobject.get_start(self)

    def get_length(self):
        start, end = self.get_start_and_end()
        return get_norm(start - end)


class Arc(TipableVMobject):
    """A circular arc."""

    def __init__(
        self,
        start_angle=0,
        angle=TAU / 4,
        radius=1.0,
        num_components=9,
        anchors_span_full_range=True,
        arc_center=ORIGIN,
        **kwargs
    ):
        if radius is None:  # apparently None is passed by ArcBetweenPoints
            radius = 1.0
        self.radius = radius
        self.num_components = num_components
        self.anchors_span_full_range = anchors_span_full_range
        self.arc_center = arc_center
        self.start_angle = start_angle
        self.angle = angle
        self._failed_to_get_center = False
        TipableVMobject.__init__(self, **kwargs)

    def generate_points(self):
        self.set_pre_positioned_points()
        self.scale(self.radius, about_point=ORIGIN)
        self.shift(self.arc_center)

    def set_pre_positioned_points(self):
        anchors = np.array(
            [
                np.cos(a) * RIGHT + np.sin(a) * UP
                for a in np.linspace(
                    self.start_angle, self.start_angle + self.angle, self.num_components
                )
            ]
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
                            smaller than half the distance between the points."""
                )
            arc_height = radius - math.sqrt(radius ** 2 - halfdist ** 2)
            angle = math.acos((radius - arc_height) / radius) * sign

        Arc.__init__(self, radius=radius, angle=angle, **kwargs)
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
        ArcBetweenPoints.__init__(self, start_point, end_point, **kwargs)
        self.add_tip(tip_shape=kwargs.pop("tip_shape", ArrowTriangleFilledTip))


class CurvedDoubleArrow(CurvedArrow):
    def __init__(self, start_point, end_point, **kwargs):
        if "tip_shape_end" in kwargs:
            kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
        tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
        CurvedArrow.__init__(self, start_point, end_point, **kwargs)
        self.add_tip(at_start=True, tip_shape=tip_shape_start)


class Circle(Arc):
    def __init__(
        self, color=RED, close_new_points=True, anchors_span_full_range=False, **kwargs
    ):
        Arc.__init__(
            self,
            start_angle=0,
            angle=TAU,
            color=color,
            close_new_points=close_new_points,
            anchors_span_full_range=anchors_span_full_range,
            **kwargs
        )

    def surround(self, mobject, dim_to_match=0, stretch=False, buffer_factor=1.2):
        # Ignores dim_to_match and stretch; result will always be a circle
        # TODO: Perhaps create an ellipse class to handle single-dimension stretching

        # Something goes wrong here when surrounding lines?
        # TODO: Figure out and fix
        self.replace(mobject, dim_to_match, stretch)

        self.width = np.sqrt(mobject.width ** 2 + mobject.height ** 2)
        return self.scale(buffer_factor)

    def point_at_angle(self, angle):
        start_angle = angle_of_vector(self.points[0] - self.get_center())
        return self.point_from_proportion((angle - start_angle) / TAU)


class Dot(Circle):
    def __init__(
        self,
        point=ORIGIN,
        radius=DEFAULT_DOT_RADIUS,
        stroke_width=0,
        fill_opacity=1.0,
        color=WHITE,
        **kwargs
    ):
        Circle.__init__(
            self,
            arc_center=point,
            radius=radius,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            color=color,
            **kwargs
        )


class SmallDot(Dot):
    """
    A dot with small radius
    """

    def __init__(self, radius=DEFAULT_SMALL_DOT_RADIUS, **kwargs):
        Dot.__init__(self, radius=radius, **kwargs)


class AnnotationDot(Dot):
    """
    A dot with bigger radius and bold stroke to annotate scenes.
    """

    def __init__(
        self,
        radius=DEFAULT_DOT_RADIUS * 1.3,
        stroke_width=5,
        stroke_color=WHITE,
        fill_color=BLUE,
        **kwargs
    ):
        Dot.__init__(
            self,
            radius=radius,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            fill_color=fill_color,
            **kwargs
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
        Dot.__init__(self, radius=radius, **kwargs)
        rendered_label.move_to(self.get_center())
        self.add(rendered_label)


class Ellipse(Circle):
    def __init__(self, width=2, height=1, **kwargs):
        Circle.__init__(self, **kwargs)
        self.stretch_to_fit_width(width)
        self.stretch_to_fit_height(height)


class AnnularSector(Arc):
    def __init__(
        self,
        inner_radius=1,
        outer_radius=2,
        angle=TAU / 4,
        start_angle=0,
        fill_opacity=1,
        stroke_width=0,
        color=WHITE,
        **kwargs
    ):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        Arc.__init__(
            self,
            start_angle=start_angle,
            angle=angle,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            color=color,
            **kwargs
        )

    def generate_points(self):
        inner_arc, outer_arc = [
            Arc(
                start_angle=self.start_angle,
                angle=self.angle,
                radius=radius,
                arc_center=self.arc_center,
            )
            for radius in (self.inner_radius, self.outer_radius)
        ]
        outer_arc.reverse_points()
        self.append_points(inner_arc.points)
        self.add_line_to(outer_arc.points[0])
        self.append_points(outer_arc.points)
        self.add_line_to(inner_arc.points[0])


class Sector(AnnularSector):
    def __init__(self, outer_radius=1, inner_radius=0, **kwargs):
        AnnularSector.__init__(
            self, inner_radius=inner_radius, outer_radius=outer_radius, **kwargs
        )


class Annulus(Circle):
    def __init__(
        self,
        inner_radius=1,
        outer_radius=2,
        fill_opacity=1,
        stroke_width=0,
        color=WHITE,
        mark_paths_closed=False,
        **kwargs
    ):
        self.mark_paths_closed = mark_paths_closed  # is this even used?
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        Circle.__init__(
            self,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            color=color,
            **kwargs
        )

    def generate_points(self):
        self.radius = self.outer_radius
        outer_circle = Circle(radius=self.outer_radius)
        inner_circle = Circle(radius=self.inner_radius)
        inner_circle.reverse_points()
        self.append_points(outer_circle.points)
        self.append_points(inner_circle.points)
        self.shift(self.arc_center)


class Line(TipableVMobject):
    def __init__(self, start=LEFT, end=RIGHT, buff=0, path_arc=None, **kwargs):
        self.buff = buff
        self.path_arc = path_arc
        self.set_start_and_end_attrs(start, end)
        TipableVMobject.__init__(self, **kwargs)

    def generate_points(self):
        if self.path_arc:
            arc = ArcBetweenPoints(self.start, self.end, angle=self.path_arc)
            self.set_points(arc.points)
        else:
            self.set_points_as_corners([self.start, self.end])
        self.account_for_buff()

    def set_path_arc(self, new_value):
        self.path_arc = new_value
        self.generate_points()

    def account_for_buff(self):
        if self.buff == 0:
            return
        #
        if self.path_arc == 0:
            length = self.get_length()
        else:
            length = self.get_arc_length()
        #
        if length < 2 * self.buff:
            return
        buff_proportion = self.buff / length
        self.pointwise_become_partial(self, buff_proportion, 1 - buff_proportion)
        return self

    def set_start_and_end_attrs(self, start, end):
        # If either start or end are Mobjects, this
        # gives their centers
        rough_start = self.pointify(start)
        rough_end = self.pointify(end)
        vect = normalize(rough_end - rough_start)
        # Now that we know the direction between them,
        # we can the appropriate boundary point from
        # start and end, if they're mobjects
        self.start = self.pointify(start, vect)
        self.end = self.pointify(end, -vect)

    def pointify(self, mob_or_point, direction=None):
        if isinstance(mob_or_point, Mobject):
            mob = mob_or_point
            if direction is None:
                return mob.get_center()
            else:
                return mob.get_boundary_point(direction)
        return np.array(mob_or_point)

    def put_start_and_end_on(self, start, end):
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

    def get_slope(self):
        return np.tan(self.get_angle())

    def set_angle(self, angle):
        return self.rotate(angle - self.get_angle(), about_point=self.get_start())

    def set_length(self, length):
        return self.scale(length / self.get_length())

    def set_opacity(self, opacity, family=True):
        # Overwrite default, which would set
        # the fill opacity
        self.set_stroke(opacity=opacity)
        if family:
            for sm in self.submobjects:
                sm.set_opacity(opacity, family)
        return self


class DashedLine(Line):
    """A dashed Line.

    Examples
    --------
    .. manim:: DashedLineExample
        :save_last_frame:

        class DashedLineExample(Scene):
            def construct(self):
                dashed_line = DashedLine(config.frame_width/2*LEFT, 4*RIGHT)
                self.add(dashed_line)

    """

    def __init__(
        self,
        *args,
        dash_length=DEFAULT_DASH_LENGTH,
        dash_spacing=None,
        positive_space_ratio=0.5,
        **kwargs
    ):
        self.dash_length = dash_length
        self.dash_spacing = (dash_spacing,)
        self.positive_space_ratio = positive_space_ratio
        Line.__init__(self, *args, **kwargs)
        dashes = DashedVMobject(
            self,
            num_dashes=self.calculate_num_dashes(positive_space_ratio),
            positive_space_ratio=positive_space_ratio,
        )
        self.clear_points()
        self.add(*dashes)

    def calculate_num_dashes(self, positive_space_ratio):
        try:
            full_length = self.dash_length / positive_space_ratio
            return int(np.ceil(self.get_length() / full_length))
        except ZeroDivisionError:
            return 1

    def calculate_positive_space_ratio(self):
        return fdiv(self.dash_length, self.dash_length + self.dash_spacing)

    def get_start(self):
        if len(self.submobjects) > 0:
            return self.submobjects[0].get_start()
        else:
            return Line.get_start(self)

    def get_end(self):
        if len(self.submobjects) > 0:
            return self.submobjects[-1].get_end()
        else:
            return Line.get_end(self)

    def get_first_handle(self):
        return self.submobjects[0].points[1]

    def get_last_handle(self):
        return self.submobjects[-1].points[-2]


class TangentLine(Line):
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


class Elbow(VMobject):
    def __init__(self, width=0.2, angle=0, **kwargs):
        self.angle = angle
        VMobject.__init__(self, **kwargs)
        self.set_points_as_corners([UP, UP + RIGHT, RIGHT])
        self.scale_to_fit_width(width, about_point=ORIGIN)
        self.rotate(self.angle, about_point=ORIGIN)


class Arrow(Line):
    def __init__(
        self,
        *args,
        stroke_width=6,
        buff=MED_SMALL_BUFF,
        max_tip_length_to_length_ratio=0.25,
        max_stroke_width_to_length_ratio=5,
        preserve_tip_size_when_scaling=True,
        **kwargs
    ):
        self.max_tip_length_to_length_ratio = (
            max_tip_length_to_length_ratio  # is this used anywhere
        )
        self.max_stroke_width_to_length_ratio = (
            max_stroke_width_to_length_ratio  # is this used anywhere
        )
        self.preserve_tip_size_when_scaling = (
            preserve_tip_size_when_scaling  # is this used anywhere
        )
        tip_shape = kwargs.pop("tip_shape", ArrowTriangleFilledTip)
        Line.__init__(self, *args, buff=buff, stroke_width=stroke_width, **kwargs)
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
            >>> scaled_arrow.get_start_and_end()
            (array([-2., -2.,  0.]), array([2., 2., 0.]))
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
            VMobject.scale(self, factor, **kwargs)
            self.set_stroke_width_from_length()
            return self

        has_tip = self.has_tip()
        has_start_tip = self.has_start_tip()
        if has_tip or has_start_tip:
            old_tips = self.pop_tips()

        VMobject.scale(self, factor, **kwargs)
        self.set_stroke_width_from_length()

        if has_tip:
            self.add_tip(tip=old_tips[0])
        if has_start_tip:
            self.add_tip(tip=old_tips[1], at_start=True)
        return self

    def get_normal_vector(self):
        p0, p1, p2 = self.tip.get_start_anchors()[:3]
        return normalize(np.cross(p2 - p1, p1 - p0))

    def reset_normal_vector(self):
        self.normal_vector = self.get_normal_vector()
        return self

    def get_default_tip_length(self):
        max_ratio = self.max_tip_length_to_length_ratio
        return min(self.tip_length, max_ratio * self.get_length())

    def set_stroke_width_from_length(self):
        max_ratio = self.max_stroke_width_to_length_ratio
        self.set_stroke(
            width=min(self.initial_stroke_width, max_ratio * self.get_length()),
            family=False,
        )
        return self


class Vector(Arrow):
    def __init__(self, direction=RIGHT, buff=0, **kwargs):
        self.buff = buff
        if len(direction) == 2:
            direction = np.append(np.array(direction), 0)
        Arrow.__init__(self, ORIGIN, direction, buff=buff, **kwargs)


class DoubleArrow(Arrow):
    def __init__(self, *args, **kwargs):
        if "tip_shape_end" in kwargs:
            kwargs["tip_shape"] = kwargs.pop("tip_shape_end")
        tip_shape_start = kwargs.pop("tip_shape_start", ArrowTriangleFilledTip)
        Arrow.__init__(self, *args, **kwargs)
        self.add_tip(at_start=True, tip_shape=tip_shape_start)


class CubicBezier(VMobject):
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
        VMobject.__init__(self, **kwargs)
        self.set_points([start_anchor, start_handle, end_handle, end_anchor])


class Polygon(VMobject):
    def __init__(self, *vertices, color=BLUE, **kwargs):
        VMobject.__init__(self, color=color, **kwargs)
        # There are actually four corners, and the first one is repeated twice to form the four vertices.
        self.set_points_as_corners([*vertices, vertices[0]])

    def get_vertices(self):
        return self.get_start_anchors()

    def round_corners(self, radius=0.5):
        vertices = self.get_vertices()
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

        self.clear_points()
        # To ensure that we loop through starting with last
        arcs = [arcs[-1], *arcs[:-1]]
        for arc1, arc2 in adjacent_pairs(arcs):
            self.append_points(arc1.points)
            line = Line(arc1.get_end(), arc2.get_start())
            # Make sure anchors are evenly distributed
            len_ratio = line.get_length() / arc1.get_arc_length()
            line.insert_n_curves(int(arc1.get_num_curves() * len_ratio))
            self.append_points(line.get_points())
        return self


class RegularPolygon(Polygon):
    def __init__(self, n=6, start_angle=None, **kwargs):
        self.start_angle = start_angle
        if self.start_angle is None:
            if n % 2 == 0:
                self.start_angle = 0
            else:
                self.start_angle = 90 * DEGREES
        start_vect = rotate_vector(RIGHT, self.start_angle)
        vertices = compass_directions(n, start_vect)
        Polygon.__init__(self, *vertices, **kwargs)


class ArcPolygon(VMobject):
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
                self.play(*[ShowCreation(ap) for ap in [ap1, ap2, ap3, ap4]])
                self.wait()

    For further examples see :class:`ArcPolygonFromArcs`.
    """

    def __init__(self, *vertices, angle=PI / 4, radius=None, arc_config=None, **kwargs):
        n = len(vertices)
        point_pairs = [(vertices[k], vertices[(k + 1) % n]) for k in range(n)]

        if not arc_config:
            if radius:
                all_arc_configs = [{"radius": radius} for pair in point_pairs]
            else:
                all_arc_configs = [{"angle": angle} for pair in point_pairs]
        elif isinstance(arc_config, dict):
            all_arc_configs = [arc_config for pair in point_pairs]
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


class ArcPolygonFromArcs(VMobject):
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
    arcpolygon itself are drawn, which affects draw time in :class:`~.ShowCreation`
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
                "All ArcPolygon submobjects must be of type Arc/ArcBetweenPoints"
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
            self.append_points(line.get_points())


class Triangle(RegularPolygon):
    def __init__(self, **kwargs):
        RegularPolygon.__init__(self, n=3, **kwargs)


class Rectangle(Polygon):
    def __init__(
        self,
        color=WHITE,
        height=2.0,
        width=4.0,
        mark_paths_closed=True,
        close_new_points=True,
        **kwargs
    ):
        self.mark_paths_closed = mark_paths_closed
        self.close_new_points = close_new_points
        Polygon.__init__(self, UL, UR, DR, DL, color=color, **kwargs)
        self.stretch_to_fit_width(width)
        self.stretch_to_fit_height(height)


class Square(Rectangle):
    def __init__(self, side_length=2.0, **kwargs):
        self.side_length = side_length
        Rectangle.__init__(self, height=side_length, width=side_length, **kwargs)


class RoundedRectangle(Rectangle):
    def __init__(self, corner_radius=0.5, **kwargs):
        self.corner_radius = corner_radius
        Rectangle.__init__(self, **kwargs)
        self.round_corners(self.corner_radius)


class ArrowTip(VMobject):
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
        ...     def __init__(self, **kwargs):
        ...         RegularPolygon.__init__(self, n=5, **kwargs)
        ...         length = 0.35
        ...         self.width = length
        ...         self.stretch_to_fit_height(length)
        >>> arr = Arrow(np.array([-2, -2, 0]), np.array([2, 2, 0]),
        ...             tip_shape=MyCustomArrowTip)
        >>> isinstance(arr.tip, RegularPolygon)
        True
        >>> from manim import Scene
        >>> class CustomTipExample(Scene):
        ...     def construct(self):
        ...         self.play(ShowCreation(arr))

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
        return get_norm(self.vector)


class ArrowTriangleTip(ArrowTip, Triangle):
    r"""Triangular arrow tip."""

    def __init__(
        self,
        fill_opacity=0,
        stroke_width=3,
        length=DEFAULT_ARROW_TIP_LENGTH,
        start_angle=PI,
        **kwargs
    ):
        Triangle.__init__(
            self,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            start_angle=start_angle,
            **kwargs
        )
        self.width = length
        self.stretch_to_fit_height(length)


class ArrowTriangleFilledTip(ArrowTriangleTip):
    r"""Triangular arrow tip with filled tip.

    This is the default arrow tip shape.
    """

    def __init__(self, fill_opacity=1, stroke_width=0, **kwargs):
        ArrowTriangleTip.__init__(
            self, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )


class ArrowCircleTip(ArrowTip, Circle):
    r"""Circular arrow tip."""

    def __init__(
        self,
        fill_opacity=0,
        stroke_width=3,
        length=DEFAULT_ARROW_TIP_LENGTH,
        start_angle=PI,
        **kwargs
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
        ArrowCircleTip.__init__(
            self, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )


class ArrowSquareTip(ArrowTip, Square):
    r"""Square arrow tip."""

    def __init__(
        self,
        fill_opacity=0,
        stroke_width=3,
        length=DEFAULT_ARROW_TIP_LENGTH,
        start_angle=PI,
        **kwargs
    ):
        self.start_angle = start_angle
        Square.__init__(
            self,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            side_length=length,
            **kwargs
        )
        self.width = length
        self.stretch_to_fit_height(length)


class ArrowSquareFilledTip(ArrowSquareTip):
    r"""Square arrow tip with filled tip."""

    def __init__(self, fill_opacity=1, stroke_width=0, **kwargs):
        ArrowSquareTip.__init__(
            self, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )


class Cutout(VMobject):
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
        VMobject.__init__(self, **kwargs)
        self.append_points(main_shape.get_points())
        if main_shape.get_direction() == "CW":
            sub_direction = "CCW"
        else:
            sub_direction = "CW"
        for mobject in mobjects:
            self.append_points(mobject.force_direction(sub_direction).get_points())
