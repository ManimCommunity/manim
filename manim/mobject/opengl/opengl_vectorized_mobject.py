from __future__ import annotations

import itertools as it
import operator as op
from collections.abc import Iterable, Sequence
from functools import reduce, wraps
from typing import Callable

import moderngl
import numpy as np

from manim import config
from manim.constants import *
from manim.mobject.opengl.opengl_mobject import OpenGLMobject, OpenGLPoint
from manim.renderer.shader_wrapper import ShaderWrapper
from manim.utils.bezier import (
    bezier,
    bezier_remap,
    get_quadratic_approximation_of_cubic,
    get_smooth_cubic_bezier_handle_points,
    integer_interpolate,
    interpolate,
    partial_bezier_points,
    proportions_along_bezier_curve_for_point,
)
from manim.utils.color import BLACK, WHITE, ManimColor, ParsableManimColor
from manim.utils.config_ops import _Data
from manim.utils.iterables import make_even, resize_with_interpolation, tuplify
from manim.utils.space_ops import (
    angle_between_vectors,
    cross2d,
    earclip_triangulation,
    get_unit_normal,
    shoelace_direction,
    z_to_vector,
)

__all__ = [
    "triggers_refreshed_triangulation",
    "OpenGLVMobject",
    "OpenGLVGroup",
    "OpenGLVectorizedPoint",
    "OpenGLCurvesAsSubmobjects",
    "OpenGLDashedVMobject",
]


def triggers_refreshed_triangulation(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        old_points = np.empty((0, 3))
        for mob in self.family_members_with_points():
            old_points = np.concatenate((old_points, mob.points), axis=0)
        func(self, *args, **kwargs)
        new_points = np.empty((0, 3))
        for mob in self.family_members_with_points():
            new_points = np.concatenate((new_points, mob.points), axis=0)
        if not np.array_equal(new_points, old_points):
            self.refresh_triangulation()
            self.refresh_unit_normal()
        return self

    return wrapper


class OpenGLVMobject(OpenGLMobject):
    """A vectorized mobject."""

    fill_dtype = [
        ("point", np.float32, (3,)),
        ("unit_normal", np.float32, (3,)),
        ("color", np.float32, (4,)),
        ("vert_index", np.float32, (1,)),
    ]
    stroke_dtype = [
        ("point", np.float32, (3,)),
        ("prev_point", np.float32, (3,)),
        ("next_point", np.float32, (3,)),
        ("unit_normal", np.float32, (3,)),
        ("stroke_width", np.float32, (1,)),
        ("color", np.float32, (4,)),
    ]
    stroke_shader_folder = "quadratic_bezier_stroke"
    fill_shader_folder = "quadratic_bezier_fill"

    fill_rgba = _Data()
    stroke_rgba = _Data()
    stroke_width = _Data()
    unit_normal = _Data()

    def __init__(
        self,
        fill_color: ParsableManimColor | None = None,
        fill_opacity: float = 0.0,
        stroke_color: ParsableManimColor | None = None,
        stroke_opacity: float = 1.0,
        stroke_width: float = DEFAULT_STROKE_WIDTH,
        draw_stroke_behind_fill: bool = False,
        # Indicates that it will not be displayed, but
        # that it should count in parent mobject's path
        pre_function_handle_to_anchor_scale_factor: float = 0.01,
        make_smooth_after_applying_functions: float = False,
        background_image_file: str | None = None,
        # This is within a pixel
        # TODO, do we care about accounting for
        # varying zoom levels?
        tolerance_for_point_equality: float = 1e-8,
        n_points_per_curve: int = 3,
        long_lines: bool = False,
        should_subdivide_sharp_curves: bool = False,
        should_remove_null_curves: bool = False,
        # Could also be "bevel", "miter", "round"
        joint_type: LineJointType | None = None,
        flat_stroke: bool = True,
        render_primitive=moderngl.TRIANGLES,
        triangulation_locked: bool = False,
        **kwargs,
    ):
        self.data = {}
        self.fill_opacity = fill_opacity
        self.stroke_opacity = stroke_opacity
        self.stroke_width = stroke_width
        self.draw_stroke_behind_fill = draw_stroke_behind_fill
        # Indicates that it will not be displayed, but
        # that it should count in parent mobject's path
        self.pre_function_handle_to_anchor_scale_factor = (
            pre_function_handle_to_anchor_scale_factor
        )
        self.make_smooth_after_applying_functions = make_smooth_after_applying_functions
        self.background_image_file = background_image_file
        # This is within a pixel
        # TODO, do we care about accounting for
        # varying zoom levels?
        self.tolerance_for_point_equality = tolerance_for_point_equality
        self.n_points_per_curve = n_points_per_curve
        self.long_lines = long_lines
        self.should_subdivide_sharp_curves = should_subdivide_sharp_curves
        self.should_remove_null_curves = should_remove_null_curves
        if joint_type is None:
            joint_type = LineJointType.AUTO
        self.joint_type = joint_type
        self.flat_stroke = flat_stroke
        self.render_primitive = render_primitive
        self.triangulation_locked = triangulation_locked

        self.needs_new_triangulation = True
        self.triangulation = np.zeros(0, dtype="i4")
        self.orientation = 1

        self.fill_data = None
        self.stroke_data = None
        self.fill_shader_wrapper = None
        self.stroke_shader_wrapper = None
        self.init_shader_data()

        super().__init__(**kwargs)
        self.refresh_unit_normal()

        if fill_color is not None:
            self.fill_color = ManimColor.parse(fill_color)
        if stroke_color is not None:
            self.stroke_color = ManimColor.parse(stroke_color)

    def _assert_valid_submobjects(self, submobjects: Iterable[OpenGLVMobject]) -> Self:
        return self._assert_valid_submobjects_internal(submobjects, OpenGLVMobject)

    def get_group_class(self):
        return OpenGLVGroup

    @staticmethod
    def get_mobject_type_class():
        return OpenGLVMobject

    def init_data(self):
        super().init_data()
        self.data.pop("rgbas")
        self.fill_rgba = np.zeros((1, 4))
        self.stroke_rgba = np.zeros((1, 4))
        self.unit_normal = np.zeros((1, 3))
        # stroke_width belongs to self.data, but is defined through init_colors+set_stroke

    # Colors
    def init_colors(self):
        self.set_fill(
            color=self.fill_color or self.color,
            opacity=self.fill_opacity,
        )
        self.set_stroke(
            color=self.stroke_color or self.color,
            width=self.stroke_width,
            opacity=self.stroke_opacity,
            background=self.draw_stroke_behind_fill,
        )
        self.set_gloss(self.gloss)
        self.set_flat_stroke(self.flat_stroke)
        return self

    def set_fill(
        self,
        color: ParsableManimColor | None = None,
        opacity: float | None = None,
        recurse: bool = True,
    ) -> OpenGLVMobject:
        """Set the fill color and fill opacity of a :class:`OpenGLVMobject`.

        Parameters
        ----------
        color
            Fill color of the :class:`OpenGLVMobject`.
        opacity
            Fill opacity of the :class:`OpenGLVMobject`.
        recurse
            If ``True``, the fill color of all submobjects is also set.

        Returns
        -------
        OpenGLVMobject
            self. For chaining purposes.

        Examples
        --------
        .. manim:: SetFill
            :save_last_frame:

            class SetFill(Scene):
                def construct(self):
                    square = Square().scale(2).set_fill(WHITE,1)
                    circle1 = Circle().set_fill(GREEN,0.8)
                    circle2 = Circle().set_fill(YELLOW) # No fill_opacity
                    circle3 = Circle().set_fill(color = '#FF2135', opacity = 0.2)
                    group = Group(circle1,circle2,circle3).arrange()
                    self.add(square)
                    self.add(group)

        See Also
        --------
        :meth:`~.OpenGLVMobject.set_style`
        """
        if opacity is not None:
            self.fill_opacity = opacity
        if recurse:
            for submobject in self.submobjects:
                submobject.set_fill(color, opacity, recurse)

        self.set_rgba_array(color, opacity, "fill_rgba", recurse)
        return self

    def set_stroke(
        self,
        color=None,
        width=None,
        opacity=None,
        background=None,
        recurse=True,
    ):
        if opacity is not None:
            self.stroke_opacity = opacity
        if recurse:
            for submobject in self.submobjects:
                submobject.set_stroke(
                    color=color,
                    width=width,
                    opacity=opacity,
                    background=background,
                    recurse=recurse,
                )

        self.set_rgba_array(color, opacity, "stroke_rgba", recurse)

        if width is not None:
            for mob in self.get_family(recurse):
                mob.stroke_width = np.array([[width] for width in tuplify(width)])

        if background is not None:
            for mob in self.get_family(recurse):
                mob.draw_stroke_behind_fill = background
        return self

    def set_style(
        self,
        fill_color=None,
        fill_opacity=None,
        fill_rgba=None,
        stroke_color=None,
        stroke_opacity=None,
        stroke_rgba=None,
        stroke_width=None,
        gloss=None,
        shadow=None,
        recurse=True,
    ):
        if fill_rgba is not None:
            self.fill_rgba = resize_with_interpolation(fill_rgba, len(fill_rgba))
        else:
            self.set_fill(color=fill_color, opacity=fill_opacity, recurse=recurse)

        if stroke_rgba is not None:
            self.stroke_rgba = resize_with_interpolation(stroke_rgba, len(fill_rgba))
            self.set_stroke(width=stroke_width)
        else:
            self.set_stroke(
                color=stroke_color,
                width=stroke_width,
                opacity=stroke_opacity,
                recurse=recurse,
            )

        if gloss is not None:
            self.set_gloss(gloss, recurse=recurse)
        if shadow is not None:
            self.set_shadow(shadow, recurse=recurse)
        return self

    def get_style(self):
        return {
            "fill_rgba": self.fill_rgba,
            "stroke_rgba": self.stroke_rgba,
            "stroke_width": self.stroke_width,
            "gloss": self.gloss,
            "shadow": self.shadow,
        }

    def match_style(self, vmobject, recurse=True):
        vmobject_style = vmobject.get_style()
        if config.renderer == RendererType.OPENGL:
            vmobject_style["stroke_width"] = vmobject_style["stroke_width"][0][0]
            vmobject_style["fill_opacity"] = self.get_fill_opacity()
        self.set_style(**vmobject_style, recurse=False)
        if recurse:
            # Does its best to match up submobject lists, and
            # match styles accordingly
            submobs1, submobs2 = self.submobjects, vmobject.submobjects
            if len(submobs1) == 0:
                return self
            elif len(submobs2) == 0:
                submobs2 = [vmobject]
            for sm1, sm2 in zip(*make_even(submobs1, submobs2)):
                sm1.match_style(sm2)
        return self

    def set_color(self, color, opacity=None, recurse=True):
        if opacity is not None:
            self.opacity = opacity

        self.set_fill(color, opacity=opacity, recurse=recurse)
        self.set_stroke(color, opacity=opacity, recurse=recurse)
        return self

    def set_opacity(self, opacity, recurse=True):
        self.set_fill(opacity=opacity, recurse=recurse)
        self.set_stroke(opacity=opacity, recurse=recurse)
        return self

    def fade(self, darkness=0.5, recurse=True):
        factor = 1.0 - darkness
        self.set_fill(
            opacity=factor * self.get_fill_opacity(),
            recurse=False,
        )
        self.set_stroke(
            opacity=factor * self.get_stroke_opacity(),
            recurse=False,
        )
        super().fade(darkness, recurse)
        return self

    # Todo im not quite sure why we are doing this
    def get_fill_colors(self):
        return [ManimColor.from_rgb(rgba[:3]) for rgba in self.fill_rgba]

    def get_fill_opacities(self):
        return self.fill_rgba[:, 3]

    def get_stroke_colors(self):
        return [ManimColor.from_rgb(rgba[:3]) for rgba in self.stroke_rgba]

    def get_stroke_opacities(self):
        return self.stroke_rgba[:, 3]

    def get_stroke_widths(self):
        return self.stroke_width

    # TODO, it's weird for these to return the first of various lists
    # rather than the full information
    def get_fill_color(self):
        """
        If there are multiple colors (for gradient)
        this returns the first one
        """
        return self.get_fill_colors()[0]

    def get_fill_opacity(self):
        """
        If there are multiple opacities, this returns the
        first
        """
        return self.get_fill_opacities()[0]

    def get_stroke_color(self):
        return self.get_stroke_colors()[0]

    def get_stroke_width(self):
        return self.get_stroke_widths()[0]

    def get_stroke_opacity(self):
        return self.get_stroke_opacities()[0]

    def get_color(self):
        if not self.has_fill():
            return self.get_stroke_color()
        return self.get_fill_color()

    def get_colors(self):
        if self.has_stroke():
            return self.get_stroke_colors()
        return self.get_fill_colors()

    stroke_color = property(get_stroke_color, set_stroke)
    color = property(get_color, set_color)
    fill_color = property(get_fill_color, set_fill)

    def has_stroke(self):
        stroke_widths = self.get_stroke_widths()
        stroke_opacities = self.get_stroke_opacities()
        return (
            stroke_widths is not None
            and stroke_opacities is not None
            and any(stroke_widths)
            and any(stroke_opacities)
        )

    def has_fill(self):
        fill_opacities = self.get_fill_opacities()
        return fill_opacities is not None and any(fill_opacities)

    def get_opacity(self):
        if self.has_fill():
            return self.get_fill_opacity()
        return self.get_stroke_opacity()

    def set_flat_stroke(self, flat_stroke=True, recurse=True):
        for mob in self.get_family(recurse):
            mob.flat_stroke = flat_stroke
        return self

    def get_flat_stroke(self):
        return self.flat_stroke

    # Points
    def set_anchors_and_handles(self, anchors1, handles, anchors2):
        assert len(anchors1) == len(handles) == len(anchors2)
        nppc = self.n_points_per_curve
        new_points = np.zeros((nppc * len(anchors1), self.dim))
        arrays = [anchors1, handles, anchors2]
        for index, array in enumerate(arrays):
            new_points[index::nppc] = array
        self.set_points(new_points)
        return self

    def start_new_path(self, point):
        assert self.get_num_points() % self.n_points_per_curve == 0
        self.append_points([point])
        return self

    def add_cubic_bezier_curve(self, anchor1, handle1, handle2, anchor2):
        new_points = get_quadratic_approximation_of_cubic(
            anchor1,
            handle1,
            handle2,
            anchor2,
        )
        self.append_points(new_points)

    def add_cubic_bezier_curve_to(self, handle1, handle2, anchor):
        """Add cubic bezier curve to the path."""
        self.throw_error_if_no_points()
        quadratic_approx = get_quadratic_approximation_of_cubic(
            self.get_last_point(),
            handle1,
            handle2,
            anchor,
        )
        if self.has_new_path_started():
            self.append_points(quadratic_approx[1:])
        else:
            self.append_points(quadratic_approx)

    def add_quadratic_bezier_curve_to(self, handle, anchor):
        self.throw_error_if_no_points()
        if self.has_new_path_started():
            self.append_points([handle, anchor])
        else:
            self.append_points([self.get_last_point(), handle, anchor])

    def add_line_to(self, point: Sequence[float]) -> OpenGLVMobject:
        """Add a straight line from the last point of OpenGLVMobject to the given point.

        Parameters
        ----------

        point
            end of the straight line.
        """
        end = self.points[-1]
        alphas = np.linspace(0, 1, self.n_points_per_curve)
        if self.long_lines:
            halfway = interpolate(end, point, 0.5)
            points = [interpolate(end, halfway, a) for a in alphas] + [
                interpolate(halfway, point, a) for a in alphas
            ]
        else:
            points = [interpolate(end, point, a) for a in alphas]
        if self.has_new_path_started():
            points = points[1:]
        self.append_points(points)
        return self

    def add_smooth_curve_to(self, point):
        if self.has_new_path_started():
            self.add_line_to(point)
        else:
            self.throw_error_if_no_points()
            new_handle = self.get_reflection_of_last_handle()
            self.add_quadratic_bezier_curve_to(new_handle, point)
        return self

    def add_smooth_cubic_curve_to(self, handle, point):
        self.throw_error_if_no_points()
        new_handle = self.get_reflection_of_last_handle()
        self.add_cubic_bezier_curve_to(new_handle, handle, point)

    def has_new_path_started(self):
        return self.get_num_points() % self.n_points_per_curve == 1

    def get_last_point(self):
        return self.points[-1]

    def get_reflection_of_last_handle(self):
        points = self.points
        return 2 * points[-1] - points[-2]

    def close_path(self):
        if not self.is_closed():
            self.add_line_to(self.get_subpaths()[-1][0])

    def is_closed(self):
        return self.consider_points_equals(self.points[0], self.points[-1])

    def subdivide_sharp_curves(self, angle_threshold=30 * DEGREES, recurse=True):
        vmobs = [vm for vm in self.get_family(recurse) if vm.has_points()]
        for vmob in vmobs:
            new_points = []
            for tup in vmob.get_bezier_tuples():
                angle = angle_between_vectors(tup[1] - tup[0], tup[2] - tup[1])
                if angle > angle_threshold:
                    n = int(np.ceil(angle / angle_threshold))
                    alphas = np.linspace(0, 1, n + 1)
                    new_points.extend(
                        [
                            partial_bezier_points(tup, a1, a2)
                            for a1, a2 in zip(alphas, alphas[1:])
                        ],
                    )
                else:
                    new_points.append(tup)
            vmob.set_points(np.vstack(new_points))
        return self

    def add_points_as_corners(self, points):
        for point in points:
            self.add_line_to(point)
        return points

    def set_points_as_corners(self, points: Iterable[float]) -> OpenGLVMobject:
        """Given an array of points, set them as corner of the vmobject.

        To achieve that, this algorithm sets handles aligned with the anchors such that the resultant bezier curve will be the segment
        between the two anchors.

        Parameters
        ----------
        points
            Array of points that will be set as corners.

        Returns
        -------
        OpenGLVMobject
            self. For chaining purposes.
        """
        nppc = self.n_points_per_curve
        points = np.array(points)
        self.set_anchors_and_handles(
            *(interpolate(points[:-1], points[1:], a) for a in np.linspace(0, 1, nppc))
        )
        return self

    def set_points_smoothly(self, points, true_smooth=False):
        self.set_points_as_corners(points)
        self.make_smooth()
        return self

    def change_anchor_mode(self, mode):
        """Changes the anchor mode of the bezier curves. This will modify the handles.

        There can be only three modes, "jagged", "approx_smooth"  and "true_smooth".

        Returns
        -------
        OpenGLVMobject
            For chaining purposes.
        """
        assert mode in ("jagged", "approx_smooth", "true_smooth")
        nppc = self.n_points_per_curve
        for submob in self.family_members_with_points():
            subpaths = submob.get_subpaths()
            submob.clear_points()
            for subpath in subpaths:
                anchors = np.vstack([subpath[::nppc], subpath[-1:]])
                new_subpath = np.array(subpath)
                if mode == "approx_smooth":
                    # TODO: get_smooth_quadratic_bezier_handle_points is not defined
                    new_subpath[1::nppc] = get_smooth_quadratic_bezier_handle_points(
                        anchors,
                    )
                elif mode == "true_smooth":
                    h1, h2 = get_smooth_cubic_bezier_handle_points(anchors)
                    new_subpath = get_quadratic_approximation_of_cubic(
                        anchors[:-1],
                        h1,
                        h2,
                        anchors[1:],
                    )
                elif mode == "jagged":
                    new_subpath[1::nppc] = 0.5 * (anchors[:-1] + anchors[1:])
                submob.append_points(new_subpath)
            submob.refresh_triangulation()
        return self

    def make_smooth(self):
        """
        This will double the number of points in the mobject,
        so should not be called repeatedly.  It also means
        transforming between states before and after calling
        this might have strange artifacts
        """
        self.change_anchor_mode("true_smooth")
        return self

    def make_approximately_smooth(self):
        """
        Unlike make_smooth, this will not change the number of
        points, but it also does not result in a perfectly smooth
        curve.  It's most useful when the points have been
        sampled at a not-too-low rate from a continuous function,
        as in the case of ParametricCurve
        """
        self.change_anchor_mode("approx_smooth")
        return self

    def make_jagged(self):
        self.change_anchor_mode("jagged")
        return self

    def add_subpath(self, points):
        assert len(points) % self.n_points_per_curve == 0
        self.append_points(points)
        return self

    def append_vectorized_mobject(self, vectorized_mobject):
        new_points = list(vectorized_mobject.points)

        if self.has_new_path_started():
            # Remove last point, which is starting
            # a new path
            self.resize_data(len(self.points - 1))
        self.append_points(new_points)
        return self

    #
    def consider_points_equals(self, p0, p1):
        return np.linalg.norm(p1 - p0) < self.tolerance_for_point_equality

    # Information about the curve
    def force_direction(self, target_direction: str):
        """Makes sure that points are either directed clockwise or
        counterclockwise.

        Parameters
        ----------
        target_direction
            Either ``"CW"`` or ``"CCW"``.
        """
        if target_direction not in ("CW", "CCW"):
            raise ValueError('Invalid input for force_direction. Use "CW" or "CCW"')

        if self.get_direction() != target_direction:
            self.reverse_points()

        return self

    def reverse_direction(self):
        """Reverts the point direction by inverting the point order.

        Returns
        -------
        :class:`OpenGLVMobject`
            Returns self.

        Examples
        --------
        .. manim:: ChangeOfDirection

            class ChangeOfDirection(Scene):
                def construct(self):
                    ccw = RegularPolygon(5)
                    ccw.shift(LEFT)
                    cw = RegularPolygon(5)
                    cw.shift(RIGHT).reverse_direction()

                    self.play(Create(ccw), Create(cw),
                    run_time=4)
        """
        self.set_points(self.points[::-1])
        return self

    def get_bezier_tuples_from_points(self, points):
        nppc = self.n_points_per_curve
        remainder = len(points) % nppc
        points = points[: len(points) - remainder]
        return points.reshape((-1, nppc, 3))

    def get_bezier_tuples(self):
        return self.get_bezier_tuples_from_points(self.points)

    def get_subpaths_from_points(self, points):
        nppc = self.n_points_per_curve
        diffs = points[nppc - 1 : -1 : nppc] - points[nppc::nppc]
        splits = (diffs * diffs).sum(1) > self.tolerance_for_point_equality
        split_indices = np.arange(nppc, len(points), nppc, dtype=int)[splits]

        # split_indices = filter(
        #     lambda n: not self.consider_points_equals(points[n - 1], points[n]),
        #     range(nppc, len(points), nppc)
        # )
        split_indices = [0, *split_indices, len(points)]
        return [
            points[i1:i2]
            for i1, i2 in zip(split_indices, split_indices[1:])
            if (i2 - i1) >= nppc
        ]

    def get_subpaths(self):
        """Returns subpaths formed by the curves of the OpenGLVMobject.

        Subpaths are ranges of curves with each pair of consecutive
        curves having their end/start points coincident.

        Returns
        -------
        Tuple
            subpaths.
        """
        return self.get_subpaths_from_points(self.points)

    def get_nth_curve_points(self, n: int) -> np.ndarray:
        """Returns the points defining the nth curve of the vmobject.

        Parameters
        ----------
        n
            index of the desired bezier curve.

        Returns
        -------
        np.ndarray
            points defininf the nth bezier curve (anchors, handles)
        """
        assert n < self.get_num_curves()
        nppc = self.n_points_per_curve
        return self.points[nppc * n : nppc * (n + 1)]

    def get_nth_curve_function(self, n: int) -> Callable[[float], np.ndarray]:
        """Returns the expression of the nth curve.

        Parameters
        ----------
        n
            index of the desired curve.

        Returns
        -------
        typing.Callable[float]
            expression of the nth bezier curve.
        """
        return bezier(self.get_nth_curve_points(n))

    def get_nth_curve_function_with_length(
        self,
        n: int,
        sample_points: int | None = None,
    ) -> tuple[Callable[[float], np.ndarray], float]:
        """Returns the expression of the nth curve along with its (approximate) length.

        Parameters
        ----------
        n
            The index of the desired curve.
        sample_points
            The number of points to sample to find the length.

        Returns
        -------
        curve : Callable[[float], np.ndarray]
            The function for the nth curve.
        length : :class:`float`
            The length of the nth curve.
        """
        if sample_points is None:
            sample_points = 10

        curve = self.get_nth_curve_function(n)
        norms = self.get_nth_curve_length_pieces(n, sample_points)

        length = np.sum(norms)

        return curve, length

    def get_num_curves(self) -> int:
        """Returns the number of curves of the vmobject.

        Returns
        -------
        int
            number of curves. of the vmobject.
        """
        return self.get_num_points() // self.n_points_per_curve

    def get_nth_curve_length(
        self,
        n: int,
        sample_points: int | None = None,
    ) -> float:
        """Returns the (approximate) length of the nth curve.

        Parameters
        ----------
        n
            The index of the desired curve.
        sample_points
            The number of points to sample to find the length.

        Returns
        -------
        length : :class:`float`
            The length of the nth curve.
        """
        _, length = self.get_nth_curve_function_with_length(n, sample_points)

        return length

    def get_curve_functions(
        self,
    ) -> Iterable[Callable[[float], np.ndarray]]:
        """Gets the functions for the curves of the mobject.

        Returns
        -------
        Iterable[Callable[[float], np.ndarray]]
            The functions for the curves.
        """
        num_curves = self.get_num_curves()

        for n in range(num_curves):
            yield self.get_nth_curve_function(n)

    def get_nth_curve_length_pieces(
        self,
        n: int,
        sample_points: int | None = None,
    ) -> np.ndarray:
        """Returns the array of short line lengths used for length approximation.

        Parameters
        ----------
        n
            The index of the desired curve.
        sample_points
            The number of points to sample to find the length.

        Returns
        -------
        np.ndarray
            The short length-pieces of the nth curve.
        """
        if sample_points is None:
            sample_points = 10

        curve = self.get_nth_curve_function(n)
        points = np.array([curve(a) for a in np.linspace(0, 1, sample_points)])
        diffs = points[1:] - points[:-1]
        norms = np.apply_along_axis(np.linalg.norm, 1, diffs)

        return norms

    def get_curve_functions_with_lengths(
        self, **kwargs
    ) -> Iterable[tuple[Callable[[float], np.ndarray], float]]:
        """Gets the functions and lengths of the curves for the mobject.

        Parameters
        ----------
        **kwargs
            The keyword arguments passed to :meth:`get_nth_curve_function_with_length`

        Returns
        -------
        Iterable[Tuple[Callable[[float], np.ndarray], float]]
            The functions and lengths of the curves.
        """
        num_curves = self.get_num_curves()

        for n in range(num_curves):
            yield self.get_nth_curve_function_with_length(n, **kwargs)

    def point_from_proportion(self, alpha: float) -> np.ndarray:
        """Gets the point at a proportion along the path of the :class:`OpenGLVMobject`.

        Parameters
        ----------
        alpha
            The proportion along the the path of the :class:`OpenGLVMobject`.

        Returns
        -------
        :class:`numpy.ndarray`
            The point on the :class:`OpenGLVMobject`.

        Raises
        ------
        :exc:`ValueError`
            If ``alpha`` is not between 0 and 1.
        :exc:`Exception`
            If the :class:`OpenGLVMobject` has no points.
        """
        if alpha < 0 or alpha > 1:
            raise ValueError(f"Alpha {alpha} not between 0 and 1.")

        self.throw_error_if_no_points()
        if alpha == 1:
            return self.points[-1]

        curves_and_lengths = tuple(self.get_curve_functions_with_lengths())

        target_length = alpha * np.sum(
            np.fromiter((length for _, length in curves_and_lengths), dtype=np.float64)
        )
        current_length = 0

        for curve, length in curves_and_lengths:
            if current_length + length >= target_length:
                if length != 0:
                    residue = (target_length - current_length) / length
                else:
                    residue = 0

                return curve(residue)

            current_length += length

    def proportion_from_point(
        self,
        point: Iterable[float | int],
    ) -> float:
        """Returns the proportion along the path of the :class:`OpenGLVMobject`
        a particular given point is at.

        Parameters
        ----------
        point
            The Cartesian coordinates of the point which may or may not lie on the :class:`OpenGLVMobject`

        Returns
        -------
        float
            The proportion along the path of the :class:`OpenGLVMobject`.

        Raises
        ------
        :exc:`ValueError`
            If ``point`` does not lie on the curve.
        :exc:`Exception`
            If the :class:`OpenGLVMobject` has no points.
        """
        self.throw_error_if_no_points()

        # Iterate over each bezier curve that the ``VMobject`` is composed of, checking
        # if the point lies on that curve. If it does not lie on that curve, add
        # the whole length of the curve to ``target_length`` and move onto the next
        # curve. If the point does lie on the curve, add how far along the curve
        # the point is to ``target_length``.
        # Then, divide ``target_length`` by the total arc length of the shape to get
        # the proportion along the ``VMobject`` the point is at.

        num_curves = self.get_num_curves()
        total_length = self.get_arc_length()
        target_length = 0
        for n in range(num_curves):
            control_points = self.get_nth_curve_points(n)
            length = self.get_nth_curve_length(n)
            proportions_along_bezier = proportions_along_bezier_curve_for_point(
                point,
                control_points,
            )
            if len(proportions_along_bezier) > 0:
                proportion_along_nth_curve = max(proportions_along_bezier)
                target_length += length * proportion_along_nth_curve
                break
            target_length += length
        else:
            raise ValueError(f"Point {point} does not lie on this curve.")

        alpha = target_length / total_length

        return alpha

    def get_anchors_and_handles(self) -> Iterable[np.ndarray]:
        """
        Returns anchors1, handles, anchors2,
        where (anchors1[i], handles[i], anchors2[i])
        will be three points defining a quadratic bezier curve
        for any i in range(0, len(anchors1))
        """
        nppc = self.n_points_per_curve
        points = self.points
        return [points[i::nppc] for i in range(nppc)]

    def get_start_anchors(self) -> np.ndarray:
        """Returns the start anchors of the bezier curves.

        Returns
        -------
        np.ndarray
            Starting anchors
        """
        return self.points[0 :: self.n_points_per_curve]

    def get_end_anchors(self) -> np.ndarray:
        """Return the starting anchors of the bezier curves.

        Returns
        -------
        np.ndarray
            Starting anchors
        """
        nppc = self.n_points_per_curve
        return self.points[nppc - 1 :: nppc]

    def get_anchors(self) -> Iterable[np.ndarray]:
        """Returns the anchors of the curves forming the OpenGLVMobject.

        Returns
        -------
        Iterable[np.ndarray]
            The anchors.
        """
        points = self.points
        if len(points) == 1:
            return points

        s = self.get_start_anchors()
        e = self.get_end_anchors()
        return list(it.chain.from_iterable(zip(s, e)))

    def get_points_without_null_curves(self, atol=1e-9):
        nppc = self.n_points_per_curve
        points = self.points
        distinct_curves = reduce(
            op.or_,
            [
                (abs(points[i::nppc] - points[0::nppc]) > atol).any(1)
                for i in range(1, nppc)
            ],
        )
        return points[distinct_curves.repeat(nppc)]

    def get_arc_length(self, sample_points_per_curve: int | None = None) -> float:
        """Return the approximated length of the whole curve.

        Parameters
        ----------
        sample_points_per_curve
            Number of sample points per curve used to approximate the length. More points result in a better approximation.

        Returns
        -------
        float
            The length of the :class:`OpenGLVMobject`.
        """
        return np.sum(
            length
            for _, length in self.get_curve_functions_with_lengths(
                sample_points=sample_points_per_curve,
            )
        )

    def get_area_vector(self):
        # Returns a vector whose length is the area bound by
        # the polygon formed by the anchor points, pointing
        # in a direction perpendicular to the polygon according
        # to the right hand rule.
        if not self.has_points():
            return np.zeros(3)

        nppc = self.n_points_per_curve
        points = self.points
        p0 = points[0::nppc]
        p1 = points[nppc - 1 :: nppc]

        # Each term goes through all edges [(x1, y1, z1), (x2, y2, z2)]
        return 0.5 * np.array(
            [
                sum(
                    (p0[:, 1] + p1[:, 1]) * (p1[:, 2] - p0[:, 2]),
                ),  # Add up (y1 + y2)*(z2 - z1)
                sum(
                    (p0[:, 2] + p1[:, 2]) * (p1[:, 0] - p0[:, 0]),
                ),  # Add up (z1 + z2)*(x2 - x1)
                sum(
                    (p0[:, 0] + p1[:, 0]) * (p1[:, 1] - p0[:, 1]),
                ),  # Add up (x1 + x2)*(y2 - y1)
            ],
        )

    def get_direction(self):
        """Uses :func:`~.space_ops.shoelace_direction` to calculate the direction.
        The direction of points determines in which direction the
        object is drawn, clockwise or counterclockwise.

        Examples
        --------
        The default direction of a :class:`~.Circle` is counterclockwise::

            >>> from manim import Circle
            >>> Circle().get_direction()
            'CCW'

        Returns
        -------
        :class:`str`
            Either ``"CW"`` or ``"CCW"``.
        """
        return shoelace_direction(self.get_start_anchors())

    def get_unit_normal(self, recompute=False):
        if not recompute:
            return self.unit_normal[0]

        if len(self.points) < 3:
            return OUT

        area_vect = self.get_area_vector()
        area = np.linalg.norm(area_vect)
        if area > 0:
            return area_vect / area
        else:
            points = self.points
            return get_unit_normal(
                points[1] - points[0],
                points[2] - points[1],
            )

    def refresh_unit_normal(self):
        for mob in self.get_family():
            mob.unit_normal[:] = mob.get_unit_normal(recompute=True)
        return self

    # Alignment
    def align_points(self, vmobject):
        # TODO: This shortcut can be a bit over eager. What if they have the same length, but different subpath lengths?
        if self.get_num_points() == len(vmobject.points):
            return

        for mob in self, vmobject:
            # If there are no points, add one to
            # where the "center" is
            if not mob.has_points():
                mob.start_new_path(mob.get_center())
            # If there's only one point, turn it into
            # a null curve
            if mob.has_new_path_started():
                mob.add_line_to(mob.points[0])

        # Figure out what the subpaths are, and align
        subpaths1 = self.get_subpaths()
        subpaths2 = vmobject.get_subpaths()
        n_subpaths = max(len(subpaths1), len(subpaths2))
        # Start building new ones
        new_subpaths1 = []
        new_subpaths2 = []

        nppc = self.n_points_per_curve

        def get_nth_subpath(path_list, n):
            if n >= len(path_list):
                # Create a null path at the very end
                return [path_list[-1][-1]] * nppc
            path = path_list[n]
            # Check for useless points at the end of the path and remove them
            # https://github.com/ManimCommunity/manim/issues/1959
            while len(path) > nppc:
                # If the last nppc points are all equal to the preceding point
                if self.consider_points_equals(path[-nppc:], path[-nppc - 1]):
                    path = path[:-nppc]
                else:
                    break
            return path

        for n in range(n_subpaths):
            sp1 = np.asarray(get_nth_subpath(subpaths1, n))
            sp2 = np.asarray(get_nth_subpath(subpaths2, n))
            diff1 = max(0, (len(sp2) - len(sp1)) // nppc)
            diff2 = max(0, (len(sp1) - len(sp2)) // nppc)
            sp1 = self.insert_n_curves_to_point_list(diff1, sp1)
            sp2 = self.insert_n_curves_to_point_list(diff2, sp2)
            new_subpaths1.append(sp1)
            new_subpaths2.append(sp2)
        self.set_points(np.vstack(new_subpaths1))
        vmobject.set_points(np.vstack(new_subpaths2))
        return self

    def insert_n_curves(self, n: int, recurse=True) -> OpenGLVMobject:
        """Inserts n curves to the bezier curves of the vmobject.

        Parameters
        ----------
        n
            Number of curves to insert.

        Returns
        -------
        OpenGLVMobject
            for chaining.
        """
        for mob in self.get_family(recurse):
            if mob.get_num_curves() > 0:
                new_points = mob.insert_n_curves_to_point_list(n, mob.points)
                # TODO, this should happen in insert_n_curves_to_point_list
                if mob.has_new_path_started():
                    new_points = np.vstack([new_points, mob.get_last_point()])
                mob.set_points(new_points)
        return self

    def insert_n_curves_to_point_list(self, n: int, points: np.ndarray) -> np.ndarray:
        """Given an array of k points defining a bezier curves
         (anchors and handles), returns points defining exactly
        k + n bezier curves.

        Parameters
        ----------
        n
            Number of desired curves.
        points
            Starting points.

        Returns
        -------
        np.ndarray
            Points generated.
        """
        nppc = self.n_points_per_curve
        if len(points) == 1:
            return np.repeat(points, nppc * n, 0)

        bezier_tuples = self.get_bezier_tuples_from_points(points)
        current_number_of_curves = len(bezier_tuples)
        new_number_of_curves = current_number_of_curves + n
        new_bezier_tuples = bezier_remap(bezier_tuples, new_number_of_curves)
        new_points = new_bezier_tuples.reshape(-1, 3)
        return new_points

    def interpolate(self, mobject1, mobject2, alpha, *args, **kwargs):
        super().interpolate(mobject1, mobject2, alpha, *args, **kwargs)
        if config["use_projection_fill_shaders"]:
            self.refresh_triangulation()
        else:
            if self.has_fill():
                tri1 = mobject1.get_triangulation()
                tri2 = mobject2.get_triangulation()
                if len(tri1) != len(tri2) or not np.all(tri1 == tri2):
                    self.refresh_triangulation()
        return self

    def pointwise_become_partial(
        self, vmobject: OpenGLVMobject, a: float, b: float, remap: bool = True
    ) -> OpenGLVMobject:
        """Given two bounds a and b, transforms the points of the self vmobject into the points of the vmobject
        passed as parameter with respect to the bounds. Points here stand for control points of the bezier curves (anchors and handles)

        Parameters
        ----------
        vmobject
            The vmobject that will serve as a model.
        a
            upper-bound.
        b
            lower-bound
        remap
            if the point amount should be kept the same (True)
            This option should be manually set to False if keeping the number of points is not needed
        """
        assert isinstance(vmobject, OpenGLVMobject)
        # Partial curve includes three portions:
        # - A middle section, which matches the curve exactly
        # - A start, which is some ending portion of an inner cubic
        # - An end, which is the starting portion of a later inner cubic
        if a <= 0 and b >= 1:
            self.set_points(vmobject.points)
            return self
        bezier_triplets = vmobject.get_bezier_tuples()
        num_quadratics = len(bezier_triplets)

        # The following two lines will compute which bezier curves of the given mobject need to be processed.
        # The residue basically indicates the proportion of the selected Bèzier curve.
        # Ex: if lower_index is 3, and lower_residue is 0.4, then the algorithm will append to the points 0.4 of the third bezier curve
        lower_index, lower_residue = integer_interpolate(0, num_quadratics, a)
        upper_index, upper_residue = integer_interpolate(0, num_quadratics, b)
        self.clear_points()
        if num_quadratics == 0:
            return self
        if lower_index == upper_index:
            self.append_points(
                partial_bezier_points(
                    bezier_triplets[lower_index],
                    lower_residue,
                    upper_residue,
                ),
            )
        else:
            self.append_points(
                partial_bezier_points(bezier_triplets[lower_index], lower_residue, 1),
            )
            inner_points = bezier_triplets[lower_index + 1 : upper_index]
            if len(inner_points) > 0:
                if remap:
                    new_triplets = bezier_remap(inner_points, num_quadratics - 2)
                else:
                    new_triplets = bezier_triplets

                self.append_points(np.asarray(new_triplets).reshape(-1, 3))
            self.append_points(
                partial_bezier_points(bezier_triplets[upper_index], 0, upper_residue),
            )
        return self

    def get_subcurve(self, a: float, b: float) -> OpenGLVMobject:
        """Returns the subcurve of the OpenGLVMobject between the interval [a, b].
        The curve is a OpenGLVMobject itself.

        Parameters
        ----------

        a
            The lower bound.
        b
            The upper bound.

        Returns
        -------
        OpenGLVMobject
            The subcurve between of [a, b]
        """
        vmob = self.copy()
        vmob.pointwise_become_partial(self, a, b)
        return vmob

    # Related to triangulation

    def refresh_triangulation(self):
        for mob in self.get_family():
            mob.needs_new_triangulation = True
        return self

    def get_triangulation(self, normal_vector=None):
        # Figure out how to triangulate the interior to know
        # how to send the points as to the vertex shader.
        # First triangles come directly from the points
        if normal_vector is None:
            normal_vector = self.get_unit_normal()

        if not self.needs_new_triangulation:
            return self.triangulation

        points = self.points

        if len(points) <= 1:
            self.triangulation = np.zeros(0, dtype="i4")
            self.needs_new_triangulation = False
            return self.triangulation

        if not np.isclose(normal_vector, OUT).all():
            # Rotate points such that unit normal vector is OUT
            points = np.dot(points, z_to_vector(normal_vector))
        indices = np.arange(len(points), dtype=int)

        b0s = points[0::3]
        b1s = points[1::3]
        b2s = points[2::3]
        v01s = b1s - b0s
        v12s = b2s - b1s

        crosses = cross2d(v01s, v12s)
        convexities = np.sign(crosses)

        atol = self.tolerance_for_point_equality
        end_of_loop = np.zeros(len(b0s), dtype=bool)
        end_of_loop[:-1] = (np.abs(b2s[:-1] - b0s[1:]) > atol).any(1)
        end_of_loop[-1] = True

        concave_parts = convexities < 0

        # These are the vertices to which we'll apply a polygon triangulation
        inner_vert_indices = np.hstack(
            [
                indices[0::3],
                indices[1::3][concave_parts],
                indices[2::3][end_of_loop],
            ],
        )
        inner_vert_indices.sort()
        rings = np.arange(1, len(inner_vert_indices) + 1)[inner_vert_indices % 3 == 2]

        # Triangulate
        inner_verts = points[inner_vert_indices]
        inner_tri_indices = inner_vert_indices[
            earclip_triangulation(inner_verts, rings)
        ]

        tri_indices = np.hstack([indices, inner_tri_indices])
        self.triangulation = tri_indices
        self.needs_new_triangulation = False
        return tri_indices

    @triggers_refreshed_triangulation
    def set_points(self, points):
        super().set_points(points)
        return self

    @triggers_refreshed_triangulation
    def set_data(self, data):
        super().set_data(data)
        return self

    # TODO, how to be smart about tangents here?
    @triggers_refreshed_triangulation
    def apply_function(self, function, make_smooth=False, **kwargs):
        super().apply_function(function, **kwargs)
        if self.make_smooth_after_applying_functions or make_smooth:
            self.make_approximately_smooth()
        return self

    @triggers_refreshed_triangulation
    def apply_points_function(self, *args, **kwargs):
        super().apply_points_function(*args, **kwargs)
        return self

    @triggers_refreshed_triangulation
    def flip(self, *args, **kwargs):
        super().flip(*args, **kwargs)
        return self

    # For shaders
    def init_shader_data(self):
        self.fill_data = np.zeros(0, dtype=self.fill_dtype)
        self.stroke_data = np.zeros(0, dtype=self.stroke_dtype)
        self.fill_shader_wrapper = ShaderWrapper(
            vert_data=self.fill_data,
            vert_indices=np.zeros(0, dtype="i4"),
            shader_folder=self.fill_shader_folder,
            render_primitive=self.render_primitive,
        )
        self.stroke_shader_wrapper = ShaderWrapper(
            vert_data=self.stroke_data,
            shader_folder=self.stroke_shader_folder,
            render_primitive=self.render_primitive,
        )

    def refresh_shader_wrapper_id(self):
        for wrapper in [self.fill_shader_wrapper, self.stroke_shader_wrapper]:
            wrapper.refresh_id()
        return self

    def get_fill_shader_wrapper(self):
        self.update_fill_shader_wrapper()
        return self.fill_shader_wrapper

    def update_fill_shader_wrapper(self):
        self.fill_shader_wrapper.vert_data = self.get_fill_shader_data()
        self.fill_shader_wrapper.vert_indices = self.get_triangulation()
        self.fill_shader_wrapper.uniforms = self.get_fill_uniforms()
        self.fill_shader_wrapper.depth_test = self.depth_test

    def get_stroke_shader_wrapper(self):
        self.update_stroke_shader_wrapper()
        return self.stroke_shader_wrapper

    def update_stroke_shader_wrapper(self):
        self.stroke_shader_wrapper.vert_data = self.get_stroke_shader_data()
        self.stroke_shader_wrapper.uniforms = self.get_stroke_uniforms()
        self.stroke_shader_wrapper.depth_test = self.depth_test

    def get_shader_wrapper_list(self):
        # Build up data lists
        fill_shader_wrappers = []
        stroke_shader_wrappers = []
        back_stroke_shader_wrappers = []
        for submob in self.family_members_with_points():
            if submob.has_fill() and not config["use_projection_fill_shaders"]:
                fill_shader_wrappers.append(submob.get_fill_shader_wrapper())
            if submob.has_stroke() and not config["use_projection_stroke_shaders"]:
                ssw = submob.get_stroke_shader_wrapper()
                if submob.draw_stroke_behind_fill:
                    back_stroke_shader_wrappers.append(ssw)
                else:
                    stroke_shader_wrappers.append(ssw)

        # Combine data lists
        wrapper_lists = [
            back_stroke_shader_wrappers,
            fill_shader_wrappers,
            stroke_shader_wrappers,
        ]
        result = []
        for wlist in wrapper_lists:
            if wlist:
                wrapper = wlist[0]
                wrapper.combine_with(*wlist[1:])
                result.append(wrapper)
        return result

    def get_stroke_uniforms(self):
        result = dict(super().get_shader_uniforms())
        result["joint_type"] = self.joint_type.value
        result["flat_stroke"] = float(self.flat_stroke)
        return result

    def get_fill_uniforms(self):
        return {
            "is_fixed_in_frame": float(self.is_fixed_in_frame),
            "is_fixed_orientation": float(self.is_fixed_orientation),
            "fixed_orientation_center": self.fixed_orientation_center,
            "gloss": self.gloss,
            "shadow": self.shadow,
        }

    def get_stroke_shader_data(self):
        points = self.points
        if len(self.stroke_data) != len(points):
            self.stroke_data = np.zeros(len(points), dtype=OpenGLVMobject.stroke_dtype)

        if "points" not in self.locked_data_keys:
            nppc = self.n_points_per_curve
            self.stroke_data["point"] = points
            self.stroke_data["prev_point"][:nppc] = points[-nppc:]
            self.stroke_data["prev_point"][nppc:] = points[:-nppc]
            self.stroke_data["next_point"][:-nppc] = points[nppc:]
            self.stroke_data["next_point"][-nppc:] = points[:nppc]

        self.read_data_to_shader(self.stroke_data, "color", "stroke_rgba")
        self.read_data_to_shader(self.stroke_data, "stroke_width", "stroke_width")
        self.read_data_to_shader(self.stroke_data, "unit_normal", "unit_normal")

        return self.stroke_data

    def get_fill_shader_data(self):
        points = self.points
        if len(self.fill_data) != len(points):
            self.fill_data = np.zeros(len(points), dtype=OpenGLVMobject.fill_dtype)
            self.fill_data["vert_index"][:, 0] = range(len(points))

        self.read_data_to_shader(self.fill_data, "point", "points")
        self.read_data_to_shader(self.fill_data, "color", "fill_rgba")
        self.read_data_to_shader(self.fill_data, "unit_normal", "unit_normal")

        return self.fill_data

    def refresh_shader_data(self):
        self.get_fill_shader_data()
        self.get_stroke_shader_data()

    def get_fill_shader_vert_indices(self):
        return self.get_triangulation()


class OpenGLVGroup(OpenGLVMobject):
    """A group of vectorized mobjects.

    This can be used to group multiple :class:`~.OpenGLVMobject` instances together
    in order to scale, move, ... them together.

    Examples
    --------

    To add :class:`~.OpenGLVMobject`s to a :class:`~.OpenGLVGroup`, you can either use the
    :meth:`~.OpenGLVGroup.add` method, or use the `+` and `+=` operators. Similarly, you
    can subtract elements of a OpenGLVGroup via :meth:`~.OpenGLVGroup.remove` method, or
    `-` and `-=` operators:

    .. doctest::

        >>> from manim import config
        >>> original_renderer = config.renderer
        >>> config.renderer = "opengl"

        >>> from manim import Triangle, Square
        >>> from manim.opengl import OpenGLVGroup
        >>> config.renderer
        <RendererType.OPENGL: 'opengl'>
        >>> vg = OpenGLVGroup()
        >>> triangle, square = Triangle(), Square()
        >>> vg.add(triangle)
        OpenGLVGroup(Triangle)
        >>> vg + square  # a new OpenGLVGroup is constructed
        OpenGLVGroup(Triangle, Square)
        >>> vg  # not modified
        OpenGLVGroup(Triangle)
        >>> vg += square  # modifies vg
        >>> vg
        OpenGLVGroup(Triangle, Square)
        >>> vg.remove(triangle)
        OpenGLVGroup(Square)
        >>> vg - square  # a new OpenGLVGroup is constructed
        OpenGLVGroup()
        >>> vg  # not modified
        OpenGLVGroup(Square)
        >>> vg -= square  # modifies vg
        >>> vg
        OpenGLVGroup()

        >>> config.renderer = original_renderer

    .. manim:: ArcShapeIris
        :save_last_frame:

        class ArcShapeIris(Scene):
            def construct(self):
                colors = [DARK_BROWN, BLUE_E, BLUE_D, BLUE_A, TEAL_B, GREEN_B, YELLOW_E]
                radius = [1 + rad * 0.1 for rad in range(len(colors))]

                circles_group = OpenGLVGroup()

                # zip(radius, color) makes the iterator [(radius[i], color[i]) for i in range(radius)]
                circles_group.add(*[Circle(radius=rad, stroke_width=10, color=col)
                                    for rad, col in zip(radius, colors)])
                self.add(circles_group)
    """

    def __init__(self, *vmobjects, **kwargs):
        super().__init__(**kwargs)
        self.add(*vmobjects)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ", ".join(str(mob) for mob in self.submobjects)
            + ")"
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__} of {len(self.submobjects)} "
            f"submobject{'s' if len(self.submobjects) > 0 else ''}"
        )

    def add(self, *vmobjects: OpenGLVMobject):
        """Checks if all passed elements are an instance of OpenGLVMobject and then add them to submobjects

        Parameters
        ----------
        vmobjects
            List of OpenGLVMobject to add

        Returns
        -------
        :class:`OpenGLVGroup`

        Raises
        ------
        TypeError
            If one element of the list is not an instance of OpenGLVMobject

        Examples
        --------
        .. manim:: AddToOpenGLVGroup

            class AddToOpenGLVGroup(Scene):
                def construct(self):
                    circle_red = Circle(color=RED)
                    circle_green = Circle(color=GREEN)
                    circle_blue = Circle(color=BLUE)
                    circle_red.shift(LEFT)
                    circle_blue.shift(RIGHT)
                    gr = OpenGLVGroup(circle_red, circle_green)
                    gr2 = OpenGLVGroup(circle_blue) # Constructor uses add directly
                    self.add(gr,gr2)
                    self.wait()
                    gr += gr2 # Add group to another
                    self.play(
                        gr.animate.shift(DOWN),
                    )
                    gr -= gr2 # Remove group
                    self.play( # Animate groups separately
                        gr.animate.shift(LEFT),
                        gr2.animate.shift(UP),
                    )
                    self.play( #Animate groups without modification
                        (gr+gr2).animate.shift(RIGHT)
                    )
                    self.play( # Animate group without component
                        (gr-circle_red).animate.shift(RIGHT)
                    )
        """
        return super().add(*vmobjects)

    def __add__(self, vmobject):
        return OpenGLVGroup(*self.submobjects, vmobject)

    def __iadd__(self, vmobject):
        return self.add(vmobject)

    def __sub__(self, vmobject):
        copy = OpenGLVGroup(*self.submobjects)
        copy.remove(vmobject)
        return copy

    def __isub__(self, vmobject):
        return self.remove(vmobject)

    def __setitem__(self, key: int, value: OpenGLVMobject | Sequence[OpenGLVMobject]):
        """Override the [] operator for item assignment.

        Parameters
        ----------
        key
            The index of the submobject to be assigned
        value
            The vmobject value to assign to the key

        Returns
        -------
        None

        Tests
        -----

        .. doctest::

            >>> from manim import config
            >>> original_renderer = config.renderer
            >>> config.renderer = "opengl"

            >>> vgroup = OpenGLVGroup(OpenGLVMobject())
            >>> new_obj = OpenGLVMobject()
            >>> vgroup[0] = new_obj

            >>> config.renderer = original_renderer
        """
        self._assert_valid_submobjects(tuplify(value))
        self.submobjects[key] = value


class OpenGLVectorizedPoint(OpenGLPoint, OpenGLVMobject):
    def __init__(
        self,
        location=ORIGIN,
        color=BLACK,
        fill_opacity=0,
        stroke_width=0,
        artificial_width=0.01,
        artificial_height=0.01,
        **kwargs,
    ):
        self.artificial_width = artificial_width
        self.artificial_height = artificial_height

        super().__init__(
            color=color, fill_opacity=fill_opacity, stroke_width=stroke_width, **kwargs
        )
        self.set_points(np.array([location]))


class OpenGLCurvesAsSubmobjects(OpenGLVGroup):
    """Convert a curve's elements to submobjects.

    Examples
    --------
    .. manim:: LineGradientExample
        :save_last_frame:

        class LineGradientExample(Scene):
            def construct(self):
                curve = ParametricFunction(lambda t: [t, np.sin(t), 0], t_range=[-PI, PI, 0.01], stroke_width=10)
                new_curve = CurvesAsSubmobjects(curve)
                new_curve.set_color_by_gradient(BLUE, RED)
                self.add(new_curve.shift(UP), curve)

    """

    def __init__(self, vmobject, **kwargs):
        super().__init__(**kwargs)
        for tup in vmobject.get_bezier_tuples():
            part = OpenGLVMobject()
            part.set_points(tup)
            part.match_style(vmobject)
            self.add(part)


class OpenGLDashedVMobject(OpenGLVMobject):
    """A :class:`OpenGLVMobject` composed of dashes instead of lines.

    Examples
    --------
    .. manim:: DashedVMobjectExample
        :save_last_frame:

        class DashedVMobjectExample(Scene):
            def construct(self):
                r = 0.5

                top_row = OpenGLVGroup()  # Increasing num_dashes
                for dashes in range(2, 12):
                    circ = DashedVMobject(Circle(radius=r, color=WHITE), num_dashes=dashes)
                    top_row.add(circ)

                middle_row = OpenGLVGroup()  # Increasing dashed_ratio
                for ratio in np.arange(1 / 11, 1, 1 / 11):
                    circ = DashedVMobject(
                        Circle(radius=r, color=WHITE), dashed_ratio=ratio
                    )
                    middle_row.add(circ)

                sq = DashedVMobject(Square(1.5, color=RED))
                penta = DashedVMobject(RegularPolygon(5, color=BLUE))
                bottom_row = OpenGLVGroup(sq, penta)

                top_row.arrange(buff=0.4)
                middle_row.arrange()
                bottom_row.arrange(buff=1)
                everything = OpenGLVGroup(top_row, middle_row, bottom_row).arrange(DOWN, buff=1)
                self.add(everything)
    """

    def __init__(
        self,
        vmobject: OpenGLVMobject,
        num_dashes: int = 15,
        dashed_ratio: float = 0.5,
        color: ParsableManimColor = WHITE,
        **kwargs,
    ):
        self.dashed_ratio = dashed_ratio
        self.num_dashes = num_dashes
        super().__init__(color=color, **kwargs)
        r = self.dashed_ratio
        n = self.num_dashes
        if num_dashes > 0:
            # Assuming total length is 1
            dash_len = r / n
            void_len = (1 - r) / n if vmobject.is_closed() else (1 - r) / (n - 1)

            self.add(
                *(
                    vmobject.get_subcurve(
                        i * (dash_len + void_len),
                        i * (dash_len + void_len) + dash_len,
                    )
                    for i in range(n)
                )
            )
        # Family is already taken care of by get_subcurve
        # implementation
        self.match_style(vmobject, recurse=False)
