from __future__ import annotations

import numpy as np

from manim.constants import *
from manim.mobject.mobject import Mobject
from manim.mobject.opengl.opengl_vectorized_mobject import (
    OpenGLDashedVMobject,
    OpenGLVGroup,
    OpenGLVMobject,
)
from manim.utils.color import *
from manim.utils.iterables import adjacent_n_tuples, adjacent_pairs
from manim.utils.simple_functions import clip
from manim.utils.space_ops import (
    angle_between_vectors,
    angle_of_vector,
    compass_directions,
    find_intersection,
    normalize,
    rotate_vector,
    rotation_matrix_transpose,
)

DEFAULT_DOT_RADIUS = 0.08
DEFAULT_DASH_LENGTH = 0.05
DEFAULT_ARROW_TIP_LENGTH = 0.35
DEFAULT_ARROW_TIP_WIDTH = 0.35

__all__ = [
    "OpenGLTipableVMobject",
    "OpenGLArc",
    "OpenGLArcBetweenPoints",
    "OpenGLCurvedArrow",
    "OpenGLCurvedDoubleArrow",
    "OpenGLCircle",
    "OpenGLDot",
    "OpenGLEllipse",
    "OpenGLAnnularSector",
    "OpenGLSector",
    "OpenGLAnnulus",
    "OpenGLLine",
    "OpenGLDashedLine",
    "OpenGLTangentLine",
    "OpenGLElbow",
    "OpenGLArrow",
    "OpenGLVector",
    "OpenGLDoubleArrow",
    "OpenGLCubicBezier",
    "OpenGLPolygon",
    "OpenGLRegularPolygon",
    "OpenGLTriangle",
    "OpenGLArrowTip",
]


class OpenGLTipableVMobject(OpenGLVMobject):
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

    # Adding, Creating, Modifying tips

    def __init__(
        self,
        tip_length=DEFAULT_ARROW_TIP_LENGTH,
        normal_vector=OUT,
        tip_config={},
        **kwargs,
    ):
        self.tip_length = tip_length
        self.normal_vector = normal_vector
        self.tip_config = tip_config
        super().__init__(**kwargs)

    def add_tip(self, at_start=False, **kwargs):
        """
        Adds a tip to the TipableVMobject instance, recognising
        that the endpoints might need to be switched if it's
        a 'starting tip' or not.
        """
        tip = self.create_tip(at_start, **kwargs)
        self.reset_endpoints_based_on_tip(tip, at_start)
        self.asign_tip_attr(tip, at_start)
        self.add(tip)
        return self

    def create_tip(self, at_start=False, **kwargs):
        """
        Stylises the tip, positions it spacially, and returns
        the newly instantiated tip to the caller.
        """
        tip = self.get_unpositioned_tip(**kwargs)
        self.position_tip(tip, at_start)
        return tip

    def get_unpositioned_tip(self, **kwargs):
        """
        Returns a tip that has been stylistically configured,
        but has not yet been given a position in space.
        """
        config = {}
        config.update(self.tip_config)
        config.update(kwargs)
        return OpenGLArrowTip(**config)

    def position_tip(self, tip, at_start=False):
        # Last two control points, defining both
        # the end, and the tangency direction
        if at_start:
            anchor = self.get_start()
            handle = self.get_first_handle()
        else:
            handle = self.get_last_handle()
            anchor = self.get_end()
        tip.rotate(angle_of_vector(handle - anchor) - PI - tip.get_angle())
        tip.shift(anchor - tip.get_tip_point())
        return tip

    def reset_endpoints_based_on_tip(self, tip, at_start):
        if self.get_length() == 0:
            # Zero length, put_start_and_end_on wouldn't
            # work
            return self

        if at_start:
            start = tip.get_base()
            end = self.get_end()
        else:
            start = self.get_start()
            end = tip.get_base()
        self.put_start_and_end_on(start, end)
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
        result = OpenGLVGroup()
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
        result = OpenGLVGroup()
        if hasattr(self, "tip"):
            result.add(self.tip)
        if hasattr(self, "start_tip"):
            result.add(self.start_tip)
        return result

    def get_tip(self):
        """Returns the TipableVMobject instance's (first) tip,
        otherwise throws an exception.
        """
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


class OpenGLArc(OpenGLTipableVMobject):
    def __init__(
        self,
        start_angle=0,
        angle=TAU / 4,
        radius=1.0,
        n_components=8,
        arc_center=ORIGIN,
        **kwargs,
    ):
        self.start_angle = start_angle
        self.angle = angle
        self.radius = radius
        self.n_components = n_components
        self.arc_center = arc_center
        super().__init__(self, **kwargs)
        self.orientation = -1

    def init_points(self):
        self.set_points(
            OpenGLArc.create_quadratic_bezier_points(
                angle=self.angle,
                start_angle=self.start_angle,
                n_components=self.n_components,
            ),
        )
        # To maintain proper orientation for fill shaders.
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

    def get_arc_center(self):
        """
        Looks at the normals to the first two
        anchors, and finds their intersection points
        """
        # First two anchors and handles
        a1, h, a2 = self.points[:3]
        # Tangent vectors
        t1 = h - a1
        t2 = h - a2
        # Normals
        n1 = rotate_vector(t1, TAU / 4)
        n2 = rotate_vector(t2, TAU / 4)
        return find_intersection(a1, n1, a2, n2)

    def get_start_angle(self):
        angle = angle_of_vector(self.get_start() - self.get_arc_center())
        return angle % TAU

    def get_stop_angle(self):
        angle = angle_of_vector(self.get_end() - self.get_arc_center())
        return angle % TAU

    def move_arc_center_to(self, point):
        self.shift(point - self.get_arc_center())
        return self


class OpenGLArcBetweenPoints(OpenGLArc):
    def __init__(self, start, end, angle=TAU / 4, **kwargs):
        super().__init__(angle=angle, **kwargs)
        if angle == 0:
            self.set_points_as_corners([LEFT, RIGHT])
        self.put_start_and_end_on(start, end)


class OpenGLCurvedArrow(OpenGLArcBetweenPoints):
    def __init__(self, start_point, end_point, **kwargs):
        super().__init__(start_point, end_point, **kwargs)
        self.add_tip()


class OpenGLCurvedDoubleArrow(OpenGLCurvedArrow):
    def __init__(self, start_point, end_point, **kwargs):
        super().__init__(start_point, end_point, **kwargs)
        self.add_tip(at_start=True)


class OpenGLCircle(OpenGLArc):
    def __init__(self, color=RED, **kwargs):
        super().__init__(0, TAU, color=color, **kwargs)

    def surround(self, mobject, dim_to_match=0, stretch=False, buff=MED_SMALL_BUFF):
        # Ignores dim_to_match and stretch; result will always be a circle
        # TODO: Perhaps create an ellipse class to handle singele-dimension stretching

        self.replace(mobject, dim_to_match, stretch)
        self.stretch((self.get_width() + 2 * buff) / self.get_width(), 0)
        self.stretch((self.get_height() + 2 * buff) / self.get_height(), 1)

    def point_at_angle(self, angle):
        start_angle = self.get_start_angle()
        return self.point_from_proportion((angle - start_angle) / TAU)


class OpenGLDot(OpenGLCircle):
    def __init__(
        self,
        point=ORIGIN,
        radius=DEFAULT_DOT_RADIUS,
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


class OpenGLEllipse(OpenGLCircle):
    def __init__(self, width=2, height=1, **kwargs):
        super().__init__(**kwargs)
        self.set_width(width, stretch=True)
        self.set_height(height, stretch=True)


class OpenGLAnnularSector(OpenGLArc):
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

    def init_points(self):
        inner_arc, outer_arc = (
            OpenGLArc(
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


class OpenGLSector(OpenGLAnnularSector):
    def __init__(self, outer_radius=1, inner_radius=0, **kwargs):
        super().__init__(inner_radius=inner_radius, outer_radius=outer_radius, **kwargs)


class OpenGLAnnulus(OpenGLCircle):
    def __init__(
        self,
        inner_radius=1,
        outer_radius=2,
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

    def init_points(self):
        self.radius = self.outer_radius
        outer_circle = OpenGLCircle(radius=self.outer_radius)
        inner_circle = OpenGLCircle(radius=self.inner_radius)
        inner_circle.reverse_points()
        self.append_points(outer_circle.points)
        self.append_points(inner_circle.points)
        self.shift(self.arc_center)


class OpenGLLine(OpenGLTipableVMobject):
    def __init__(self, start=LEFT, end=RIGHT, buff=0, path_arc=0, **kwargs):
        self.dim = 3
        self.buff = buff
        self.path_arc = path_arc
        self.set_start_and_end_attrs(start, end)
        super().__init__(**kwargs)

    def init_points(self):
        self.set_points_by_ends(self.start, self.end, self.buff, self.path_arc)

    def set_points_by_ends(self, start, end, buff=0, path_arc=0):
        if path_arc:
            self.set_points(OpenGLArc.create_quadratic_bezier_points(path_arc))
            self.put_start_and_end_on(start, end)
        else:
            self.set_points_as_corners([start, end])
        self.account_for_buff(self.buff)

    def set_path_arc(self, new_value):
        self.path_arc = new_value
        self.init_points()

    def account_for_buff(self, buff):
        if buff == 0:
            return
        #
        length = self.get_length() if self.path_arc == 0 else self.get_arc_length()
        #
        if length < 2 * buff:
            return
        buff_prop = buff / length
        self.pointwise_become_partial(self, buff_prop, 1 - buff_prop)
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
        self.start = self.pointify(start, vect) + self.buff * vect
        self.end = self.pointify(end, -vect) - self.buff * vect

    def pointify(self, mob_or_point, direction=None):
        """
        Take an argument passed into Line (or subclass) and turn
        it into a 3d point.
        """
        if isinstance(mob_or_point, Mobject):
            mob = mob_or_point
            if direction is None:
                return mob.get_center()
            else:
                return mob.get_continuous_bounding_box_point(direction)
        else:
            point = mob_or_point
            result = np.zeros(self.dim)
            result[: len(point)] = point
            return result

    def put_start_and_end_on(self, start, end):
        curr_start, curr_end = self.get_start_and_end()
        if (curr_start == curr_end).all():
            self.set_points_by_ends(start, end, self.path_arc)
        return super().put_start_and_end_on(start, end)

    def get_vector(self):
        return self.get_end() - self.get_start()

    def get_unit_vector(self):
        return normalize(self.get_vector())

    def get_angle(self):
        return angle_of_vector(self.get_vector())

    def get_projection(self, point):
        """Return projection of a point onto the line"""
        unit_vect = self.get_unit_vector()
        start = self.get_start()
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
        self.scale(length / self.get_length())


class OpenGLDashedLine(OpenGLLine):
    def __init__(
        self, *args, dash_length=DEFAULT_DASH_LENGTH, dashed_ratio=0.5, **kwargs
    ):
        self.dashed_ratio = dashed_ratio
        self.dash_length = dash_length
        super().__init__(*args, **kwargs)
        dashed_ratio = self.dashed_ratio
        num_dashes = self.calculate_num_dashes(dashed_ratio)
        dashes = OpenGLDashedVMobject(
            self,
            num_dashes=num_dashes,
            dashed_ratio=dashed_ratio,
        )
        self.clear_points()
        self.add(*dashes)

    def calculate_num_dashes(self, dashed_ratio):
        return max(
            2,
            int(np.ceil((self.get_length() / self.dash_length) * dashed_ratio)),
        )

    def get_start(self):
        if len(self.submobjects) > 0:
            return self.submobjects[0].get_start()
        else:
            return super().get_start()

    def get_end(self):
        if len(self.submobjects) > 0:
            return self.submobjects[-1].get_end()
        else:
            return super().get_end()

    def get_first_handle(self):
        return self.submobjects[0].points[1]

    def get_last_handle(self):
        return self.submobjects[-1].points[-2]


class OpenGLTangentLine(OpenGLLine):
    def __init__(self, vmob, alpha, length=1, d_alpha=1e-6, **kwargs):
        self.length = length
        self.d_alpha = d_alpha
        da = self.d_alpha
        a1 = clip(alpha - da, 0, 1)
        a2 = clip(alpha + da, 0, 1)
        super().__init__(vmob.pfp(a1), vmob.pfp(a2), **kwargs)
        self.scale(self.length / self.get_length())


class OpenGLElbow(OpenGLVMobject):
    def __init__(self, width=0.2, angle=0, **kwargs):
        self.angle = angle
        super().__init__(self, **kwargs)
        self.set_points_as_corners([UP, UP + RIGHT, RIGHT])
        self.set_width(width, about_point=ORIGIN)
        self.rotate(self.angle, about_point=ORIGIN)


class OpenGLArrow(OpenGLLine):
    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        path_arc=0,
        fill_color=GREY_A,
        fill_opacity=1,
        stroke_width=0,
        buff=MED_SMALL_BUFF,
        thickness=0.05,
        tip_width_ratio=5,
        tip_angle=PI / 3,
        max_tip_length_to_length_ratio=0.5,
        max_width_to_length_ratio=0.1,
        **kwargs,
    ):
        self.thickness = thickness
        self.tip_width_ratio = tip_width_ratio
        self.tip_angle = tip_angle
        self.max_tip_length_to_length_ratio = max_tip_length_to_length_ratio
        self.max_width_to_length_ratio = max_width_to_length_ratio
        super().__init__(
            start=start,
            end=end,
            buff=buff,
            path_arc=path_arc,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs,
        )

    def set_points_by_ends(self, start, end, buff=0, path_arc=0):
        # Find the right tip length and thickness
        vect = end - start
        length = max(np.linalg.norm(vect), 1e-8)
        thickness = self.thickness
        w_ratio = self.max_width_to_length_ratio / (thickness / length)
        if w_ratio < 1:
            thickness *= w_ratio

        tip_width = self.tip_width_ratio * thickness
        tip_length = tip_width / (2 * np.tan(self.tip_angle / 2))
        t_ratio = self.max_tip_length_to_length_ratio / (tip_length / length)
        if t_ratio < 1:
            tip_length *= t_ratio
            tip_width *= t_ratio

        # Find points for the stem
        if path_arc == 0:
            points1 = (length - tip_length) * np.array([RIGHT, 0.5 * RIGHT, ORIGIN])
            points1 += thickness * UP / 2
            points2 = points1[::-1] + thickness * DOWN
        else:
            # Solve for radius so that the tip-to-tail length matches |end - start|
            a = 2 * (1 - np.cos(path_arc))
            b = -2 * tip_length * np.sin(path_arc)
            c = tip_length**2 - length**2
            R = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

            # Find arc points
            points1 = OpenGLArc.create_quadratic_bezier_points(path_arc)
            points2 = np.array(points1[::-1])
            points1 *= R + thickness / 2
            points2 *= R - thickness / 2
            if path_arc < 0:
                tip_length *= -1
            rot_T = rotation_matrix_transpose(PI / 2 - path_arc, OUT)
            for points in points1, points2:
                points[:] = np.dot(points, rot_T)
                points += R * DOWN

        self.set_points(points1)
        # Tip
        self.add_line_to(tip_width * UP / 2)
        self.add_line_to(tip_length * LEFT)
        self.tip_index = len(self.points) - 1
        self.add_line_to(tip_width * DOWN / 2)
        self.add_line_to(points2[0])
        # Close it out
        self.append_points(points2)
        self.add_line_to(points1[0])

        if length > 0:
            # Final correction
            super().scale(length / self.get_length())

        self.rotate(angle_of_vector(vect) - self.get_angle())
        self.rotate(
            PI / 2 - np.arccos(normalize(vect)[2]),
            axis=rotate_vector(self.get_unit_vector(), -PI / 2),
        )
        self.shift(start - self.get_start())
        self.refresh_triangulation()

    def reset_points_around_ends(self):
        self.set_points_by_ends(
            self.get_start(),
            self.get_end(),
            path_arc=self.path_arc,
        )
        return self

    def get_start(self):
        nppc = self.n_points_per_curve
        points = self.points
        return (points[0] + points[-nppc]) / 2

    def get_end(self):
        return self.points[self.tip_index]

    def put_start_and_end_on(self, start, end):
        self.set_points_by_ends(start, end, buff=0, path_arc=self.path_arc)
        return self

    def scale(self, *args, **kwargs):
        super().scale(*args, **kwargs)
        self.reset_points_around_ends()
        return self

    def set_thickness(self, thickness):
        self.thickness = thickness
        self.reset_points_around_ends()
        return self

    def set_path_arc(self, path_arc):
        self.path_arc = path_arc
        self.reset_points_around_ends()
        return self


class OpenGLVector(OpenGLArrow):
    def __init__(self, direction=RIGHT, buff=0, **kwargs):
        self.buff = buff
        if len(direction) == 2:
            direction = np.hstack([direction, 0])
        super().__init__(ORIGIN, direction, buff=buff, **kwargs)


class OpenGLDoubleArrow(OpenGLArrow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_tip(at_start=True)


class OpenGLCubicBezier(OpenGLVMobject):
    def __init__(self, a0, h0, h1, a1, **kwargs):
        super().__init__(**kwargs)
        self.add_cubic_bezier_curve(a0, h0, h1, a1)


class OpenGLPolygon(OpenGLVMobject):
    def __init__(self, *vertices, **kwargs):
        self.vertices = vertices
        super().__init__(**kwargs)

    def init_points(self):
        verts = self.vertices
        self.set_points_as_corners([*verts, verts[0]])

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
            arc = OpenGLArcBetweenPoints(
                v2 - unit_vect1 * cut_off_length,
                v2 + unit_vect2 * cut_off_length,
                angle=sign * angle,
                n_components=2,
            )
            arcs.append(arc)

        self.clear_points()
        # To ensure that we loop through starting with last
        arcs = [arcs[-1], *arcs[:-1]]
        for arc1, arc2 in adjacent_pairs(arcs):
            self.append_points(arc1.points)
            line = OpenGLLine(arc1.get_end(), arc2.get_start())
            # Make sure anchors are evenly distributed
            len_ratio = line.get_length() / arc1.get_arc_length()
            line.insert_n_curves(int(arc1.get_num_curves() * len_ratio))
            self.append_points(line.points)
        return self


class OpenGLRegularPolygon(OpenGLPolygon):
    def __init__(self, n=6, start_angle=None, **kwargs):
        self.start_angle = start_angle
        if self.start_angle is None:
            if n % 2 == 0:
                self.start_angle = 0
            else:
                self.start_angle = 90 * DEGREES
        start_vect = rotate_vector(RIGHT, self.start_angle)
        vertices = compass_directions(n, start_vect)
        super().__init__(*vertices, **kwargs)


class OpenGLTriangle(OpenGLRegularPolygon):
    def __init__(self, **kwargs):
        super().__init__(n=3, **kwargs)


class OpenGLArrowTip(OpenGLTriangle):
    def __init__(
        self,
        fill_opacity=1,
        fill_color=WHITE,
        stroke_width=0,
        width=DEFAULT_ARROW_TIP_WIDTH,
        length=DEFAULT_ARROW_TIP_LENGTH,
        angle=0,
        **kwargs,
    ):
        super().__init__(
            start_angle=0,
            fill_opacity=fill_opacity,
            fill_color=fill_color,
            stroke_width=stroke_width,
            **kwargs,
        )
        self.set_width(width, stretch=True)
        self.set_height(length, stretch=True)

    def get_base(self):
        return self.point_from_proportion(0.5)

    def get_tip_point(self):
        return self.points[0]

    def get_vector(self):
        return self.get_tip_point() - self.get_base()

    def get_angle(self):
        return angle_of_vector(self.get_vector())

    def get_length(self):
        return np.linalg.norm(self.get_vector())


class OpenGLRectangle(OpenGLPolygon):
    def __init__(self, color=WHITE, width=4.0, height=2.0, **kwargs):
        super().__init__(UR, UL, DL, DR, color=color, **kwargs)

        self.set_width(width, stretch=True)
        self.set_height(height, stretch=True)


class OpenGLSquare(OpenGLRectangle):
    def __init__(self, side_length=2.0, **kwargs):
        self.side_length = side_length

        super().__init__(height=side_length, width=side_length, **kwargs)


class OpenGLRoundedRectangle(OpenGLRectangle):
    def __init__(self, corner_radius=0.5, **kwargs):
        self.corner_radius = corner_radius
        super().__init__(**kwargs)
        self.round_corners(self.corner_radius)
