from __future__ import annotations

import itertools as it
import operator as op
from functools import reduce
from typing import TYPE_CHECKING, Literal

import numpy as np
from typing_extensions import Unpack

from manim.constants import *
from manim.mobject.opengl.opengl_mobject import (
    MobjectKwargs,
    OpenGLMobject,
    OpenGLPoint,
)
from manim.utils.bezier import (
    bezier,
    bezier_remap,
    get_quadratic_approximation_of_cubic,
    get_smooth_cubic_bezier_handle_points,
    get_smooth_quadratic_bezier_handle_points,
    integer_interpolate,
    interpolate,
    partial_bezier_points,
    proportions_along_bezier_curve_for_point,
)
from manim.utils.color import *
from manim.utils.color.core import ParsableManimColor
from manim.utils.deprecation import deprecated
from manim.utils.iterables import (
    listify,
    make_even,
)
from manim.utils.space_ops import (
    angle_between_vectors,
    get_norm,
    get_unit_normal,
    shoelace_direction,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import Self

    import numpy.typing as npt

    from manim.typing import Point3D, Point3DLike, Point3DLike_Array

__all__ = [
    "OpenGLVMobject",
    "OpenGLVGroup",
    "OpenGLVectorizedPoint",
    "OpenGLCurvesAsSubmobjects",
    "OpenGLDashedVMobject",
]


DEFAULT_STROKE_COLOR = GREY_A
DEFAULT_FILL_COLOR = GREY_C


# TODO: add this to the **kwargs of all mobjects that use OpenGLVMobject
class VMobjectKwargs(MobjectKwargs, total=False):
    color: ParsableManimColor | Sequence[ParsableManimColor] | None
    fill_color: ParsableManimColor | Sequence[ParsableManimColor] | None
    fill_opacity: float | None
    stroke_color: ParsableManimColor | Sequence[ParsableManimColor] | None
    stroke_opacity: float | None
    stroke_width: float
    draw_stroke_behind_fill: bool
    background_image_file: str | None
    long_lines: bool
    joint_type: LineJointType
    flat_stroke: bool
    shade_in_3d: bool
    checkerboard_colors: bool  # TODO: remove


class OpenGLVMobject(OpenGLMobject):
    """A vectorized mobject."""

    n_points_per_curve: int = 3
    pre_function_handle_to_anchor_scale_factor: float = 0.01
    make_smooth_after_applying_functions: bool = False
    tolerance_for_point_equality: float = 1e-8

    # WARNING: before updating the __init__ update the VMobjectKwargs TypedDict
    # so users can get autocomplete
    def __init__(
        self,
        fill_color: ParsableManimColor | Sequence[ParsableManimColor] | None = None,
        fill_opacity: float | None = None,
        stroke_color: ParsableManimColor | Sequence[ParsableManimColor] | None = None,
        stroke_opacity: float | None = None,
        stroke_width: float = DEFAULT_STROKE_WIDTH,
        draw_stroke_behind_fill: bool = False,
        background_image_file: str | None = None,
        long_lines: bool = False,
        joint_type: LineJointType = LineJointType.AUTO,
        flat_stroke: bool = False,
        shade_in_3d: bool = False,  # TODO: Can be ignored for now but we should think about using some sort of shader to introduce lighting after deferred rendering has completed
        checkerboard_colors: bool = False,  # ignore,
        **kwargs: Unpack[MobjectKwargs],
    ):
        self.stroke_width = listify(stroke_width)
        self.draw_stroke_behind_fill = draw_stroke_behind_fill
        self.background_image_file = background_image_file
        self.long_lines = long_lines
        self.joint_type = joint_type
        self.flat_stroke = flat_stroke

        self.needs_new_triangulation = True
        self.triangulation = np.zeros(0, dtype="i4")

        super().__init__(**kwargs)
        if fill_color is None:
            fill_color = self.color
        if stroke_color is None:
            stroke_color = self.color
        self.set_fill(color=fill_color, opacity=fill_opacity)
        self.set_stroke(color=stroke_color, width=stroke_width, opacity=stroke_opacity)

        # self.refresh_unit_normal()

    def _assert_valid_submobjects(self, submobjects: Iterable[OpenGLVMobject]) -> Self:
        return self._assert_valid_submobjects_internal(submobjects, OpenGLVMobject)

    def get_group_class(self) -> type[OpenGLVGroup]:  # type: ignore
        return OpenGLVGroup

    @staticmethod
    def get_mobject_type_class():
        return OpenGLVMobject

    # These are here just to make type checkers happy
    def get_family(self, recurse: bool = True) -> Sequence[OpenGLVMobject]:
        return super().get_family(recurse)  # type: ignore

    def family_members_with_points(self) -> Sequence[OpenGLVMobject]:  # type: ignore
        return super().family_members_with_points()  # type: ignore

    def replicate(self, n: int) -> OpenGLVGroup:  # type: ignore
        return super().replicate(n)  # type: ignore

    def get_grid(self, *args, **kwargs) -> OpenGLVGroup:  # type: ignore
        return super().get_grid(*args, **kwargs)  # type: ignore

    def __getitem__(self, value: int | slice) -> Self:  # type: ignore
        return super().__getitem__(value)  # type: ignore

    def add(self, *vmobjects: OpenGLVMobject) -> Self:  # type: ignore
        return super().add(*vmobjects)

    # Colors
    def init_colors(self) -> Self:
        # self.set_fill(
        #     color=self.fill_color or self.color,
        #     opacity=self.fill_opacity,
        # )
        # self.set_stroke(
        #     color=self.stroke_color or self.color,
        #     width=self.stroke_width,
        #     opacity=self.stroke_opacity,
        #     background=self.draw_stroke_behind_fill,
        # )
        # self.set_gloss(self.gloss)
        # self.set_flat_stroke(self.flat_stroke)
        # self.color = self.get_color()
        return self

    def set_fill(
        self,
        color: ParsableManimColor | Sequence[ParsableManimColor] | None = None,
        opacity: float | None = None,
        recurse: bool = True,
    ) -> Self:
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
        if recurse:
            for submob in self.submobjects:
                submob.set_fill(color, opacity, recurse=True)
        if color is not None:
            self.fill_color: list[ManimColor] = listify(ManimColor.parse(color))
        if opacity is not None:
            self.fill_color = [c.opacity(opacity) for c in self.fill_color]
        return self

    def set_stroke(
        self,
        color=None,
        width=None,
        opacity=None,
        background=None,
        recurse=True,
    ):
        for mob in self.get_family(recurse):
            if color is not None:
                mob.stroke_color = listify(ManimColor.parse(color))
            if opacity is not None:
                mob.stroke_color = [c.opacity(opacity) for c in mob.stroke_color]

            if width is not None:
                mob.stroke_width = listify(width)

            if background is not None:
                mob.draw_stroke_behind_fill = background
        return self

    def set_backstroke(
        self,
        color: ManimColor | Iterable[ManimColor] | None = None,
        width: float | Iterable[float] = 3,
        background: bool = True,
    ) -> Self:
        self.set_stroke(color, width, background=background)
        return self

    def set_style(
        self,
        fill_color: ParsableManimColor | Iterable[ParsableManimColor] | None = None,
        fill_opacity: float | None = None,
        stroke_color: ParsableManimColor | Iterable[ParsableManimColor] | None = None,
        stroke_opacity: float | Iterable[float] | None = None,
        stroke_width: float | Iterable[float] | None = None,
        stroke_background: bool = True,
        reflectiveness: float | None = None,
        gloss: float | None = None,
        shadow: float | None = None,
        recurse: bool = True,
    ) -> Self:
        for mob in self.get_family(recurse):
            mob.set_fill(color=fill_color, opacity=fill_opacity, recurse=False)
            mob.set_stroke(
                color=stroke_color,
                width=stroke_width,
                opacity=stroke_opacity,
                recurse=False,
                background=stroke_background,
            )
        return self

    def get_style(self):
        return {
            "fill_color": self.fill_color.copy(),
            "stroke_color": self.stroke_color.copy(),
            "stroke_width": self.stroke_width.copy(),
            # "stroke_background": self.draw_stroke_behind_fill,
        }

    def match_style(self, vmobject: OpenGLVMobject, recurse: bool = True):
        self.set_style(**vmobject.get_style(), recurse=False)
        if recurse:
            # Does its best to match up submobject lists, and
            # match styles accordingly
            submobs1, submobs2 = self.submobjects, vmobject.submobjects
            if len(submobs1) == 0:
                return self
            elif len(submobs2) == 0:
                submobs2 = [vmobject]
            for sm1, sm2 in zip(*make_even(submobs1, submobs2), strict=False):
                sm1.match_style(sm2)
        return self

    def set_color(self, color, opacity=None, recurse=True) -> Self:
        self.set_fill(color, opacity=opacity, recurse=recurse)
        self.set_stroke(color, opacity=opacity, recurse=recurse)
        return self

    def set_opacity(self, opacity, recurse=True) -> Self:
        self.set_fill(opacity=opacity, recurse=recurse)
        self.set_stroke(opacity=opacity, recurse=recurse)
        return self

    def fade(self, darkness=0.5, recurse=True) -> Self:
        mobs = self.get_family() if recurse else [self]
        for mob in mobs:
            factor = 1.0 - darkness
            mob.set_fill(
                opacity=factor * mob.get_fill_opacity(),
                recurse=False,
            )
            mob.set_stroke(
                opacity=factor * mob.get_stroke_opacity(),
                recurse=False,
            )
        return self

    # Todo im not quite sure why we are doing this
    def get_fill_colors(self):
        return self.fill_color

    def get_fill_opacities(self) -> np.ndarray:
        return [c.to_rgba()[3] for c in self.fill_color]

    def get_stroke_colors(self):
        return self.stroke_color

    def get_stroke_opacities(self) -> np.ndarray:
        return [c.to_rgba()[3] for c in self.stroke_color]

    def get_stroke_widths(self) -> np.ndarray:
        return self.stroke_width

    # TODO, it's weird for these to return the first of various lists
    # rather than the full information
    def get_fill_color(self) -> ManimColor:
        """
        If there are multiple colors (for gradient)
        this returns the first one
        """
        return self.get_fill_colors()[0]

    def get_fill_opacity(self) -> float:
        """
        If there are multiple opacities, this returns the
        first
        """
        return self.get_fill_opacities()[0]

    def get_stroke_color(self) -> ManimColor:
        return self.get_stroke_colors()[0]

    def get_stroke_width(self) -> float | np.ndarray:
        return self.get_stroke_widths()[0]

    def get_stroke_opacity(self) -> float:
        return self.get_stroke_opacities()[0]

    def get_color(self) -> ManimColor:
        if self.has_fill():
            return self.get_fill_color()
        return self.get_stroke_color()

    def has_stroke(self) -> bool:
        # TODO: This currently doesn't make sense needs fixing
        return len(self.stroke_width) > 0 and any(self.get_stroke_opacities())

    def has_fill(self) -> bool:
        return any(self.get_fill_opacities())

    def get_opacity(self) -> float:
        if self.has_fill():
            return self.get_fill_opacity()
        return self.get_stroke_opacity()

    def set_flat_stroke(self, flat_stroke: bool = True, recurse: bool = True):
        for mob in self.get_family(recurse):
            mob.uniforms["flat_stroke"] = float(flat_stroke)
        return self

    def get_flat_stroke(self) -> bool:
        return self.uniforms["flat_stroke"] == 1.0

    def set_joint_type(self, joint_type: LineJointType, recurse: bool = True):
        for mob in self.get_family(recurse):
            mob.uniforms["joint_type"] = float(joint_type.value)
        return self

    def get_joint_type(self) -> LineJointType:
        return LineJointType(int(self.uniforms["joint_type"]))

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

    def add_cubic_bezier_curve(
        self,
        anchor1: Point3DLike,
        handle1: Point3DLike,
        handle2: Point3DLike,
        anchor2: Point3DLike,
    ):
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

    def add_line_to(self, point: Sequence[float] | npt.NDArray[float]) -> Self:
        """Add a straight line from the last point of OpenGLVMobject to the given point.

        Parameters
        ----------

        point
            end of the straight line.
        """
        point = np.asarray(point)
        if not self.has_points():
            self.points = np.array([point])
            return self
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

    def add_smooth_cubic_curve_to(self, handle: np.ndarray, point: np.ndarray):
        self.throw_error_if_no_points()
        if self.get_num_points() == 1:
            new_handle = self.points[-1]
        else:
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
        vmobs = [vm for vm in self.get_family(recurse=recurse) if vm.has_points()]
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
                            for a1, a2 in zip(alphas, alphas[1:], strict=False)
                        ],
                    )
                else:
                    new_points.append(tup)
            vmob.set_points(np.vstack(new_points))
        return self

    def add_points_as_corners(self, points):
        for point in points:
            self.add_line_to(point)
        return self

    def set_points_as_corners(self, points: Point3DLike_Array) -> OpenGLVMobject:
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

    def set_points_smoothly(
        self, points: Point3DLike_Array, true_smooth: bool = False
    ) -> Self:
        self.set_points_as_corners(points)
        if true_smooth:
            self.make_smooth()
        else:
            self.make_approximately_smooth()
        return self

    def change_anchor_mode(
        self, mode: Literal["jagged", "approx_smooth", "true_smooth"]
    ) -> Self:
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
                    new_subpath[1::nppc] = get_smooth_quadratic_bezier_handle_points(
                        anchors
                    )
                elif mode == "true_smooth":
                    h1, h2 = get_smooth_cubic_bezier_handle_points(anchors)
                    new_subpath = get_quadratic_approximation_of_cubic(
                        anchors[:-1], h1, h2, anchors[1:]
                    )
                elif mode == "jagged":
                    new_subpath[1::nppc] = 0.5 * (anchors[:-1] + anchors[1:])
                submob.append_points(new_subpath)
            # TODO: not implemented
            # submob.refresh_triangulation()
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
            self.points = self.points[:-1]
        self.append_points(new_points)
        return self

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
            for i1, i2 in zip(split_indices, split_indices[1:], strict=False)
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

    def quick_point_from_proportion(self, alpha: float) -> np.ndarray:
        # Assumes all curves have the same length, so is inaccurate
        num_curves = self.get_num_curves()
        n, residue = integer_interpolate(0, num_curves, alpha)
        curve_func = self.get_nth_curve_function(n)
        return curve_func(residue)

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
        norms = np.apply_along_axis(np.linalg.norm, 1, diffs)  # type: ignore

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

    def point_from_proportion(self, alpha: float) -> Point3D:
        """Gets the point at a proportion along the path of the :class:`OpenGLVMobject`.

        Parameters
        ----------
        alpha
            The proportion along the the path of the :class:`OpenGLVMobject`.

        Returns
        -------
        :class:`Point3D`
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
        point: Point3DLike,
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
        target_length = 0.0
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
        return list(it.chain.from_iterable(zip(s, e, strict=False)))

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

    def get_arc_length(self, n_sample_points: int | None = None) -> float:
        """Return the approximated length of the whole curve.

        Parameters
        ----------
        n_sample_points
            The number of points to sample. If ``None``, the number of points is calculated automatically.
            Takes points on the outline of the :class:`OpenGLVMobject` and calculates the distance between them.

        Returns
        -------
        float
            The length of the :class:`OpenGLVMobject`.
        """
        if n_sample_points is None:
            n_sample_points = 4 * self.get_num_curves() + 1
        points = np.array(
            [self.point_from_proportion(a) for a in np.linspace(0, 1, n_sample_points)]
        )
        diffs = points[1:] - points[:-1]
        norms = np.array([get_norm(d) for d in diffs])
        return norms.sum()

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

        if len(p0) != len(p1):
            m = min(len(p0), len(p1))
            p0 = p0[:m]
            p1 = p1[:m]

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

    def get_unit_normal(self) -> np.ndarray:
        if self.get_num_points() < 3:
            return OUT

        area_vect = self.get_area_vector()
        area = get_norm(area_vect)
        if area > 0:
            normal = area_vect / area
        else:
            points = self.points
            normal = get_unit_normal(
                points[1] - points[0],
                points[2] - points[1],
            )
        return normal

    # Alignment
    def align_points(self, vmobject: OpenGLVMobject) -> Self:
        # TODO: This shortcut can be a bit over eager. What if they have the same length, but different subpath lengths?
        if self.get_num_points() == vmobject.get_num_points():
            return

        for mob in (self, vmobject):
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
            # # Check for useless points at the end of the path and remove them
            # # https://github.com/ManimCommunity/manim/issues/1959
            # while len(path) > nppc:
            #     # If the last nppc points are all equal to the preceding point
            #     if self.consider_points_equals(path[-nppc:], path[-nppc - 1]):
            #         path = path[:-nppc]
            #     else:
            #         break
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

    def insert_n_curves(self, n: int, recurse=True) -> Self:
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
        """Given an array of 3k points defining a Bézier curve (anchors and
        handles), return 3(k+n) points defining exactly k + n Bézier curves.

        Parameters
        ----------
        n
            Number of desired curves to insert.
        points
            Starting points.

        Returns
        -------
        np.ndarray
            Points generated.
        """
        if len(points) == 1:
            nppc = self.n_points_per_curve
            return np.repeat(points, nppc * n, 0)
        bezier_tuples = self.get_bezier_tuples_from_points(points)
        current_number_of_curves = len(bezier_tuples)
        new_number_of_curves = current_number_of_curves + n
        new_bezier_tuples = bezier_remap(bezier_tuples, new_number_of_curves)
        new_points = new_bezier_tuples.reshape(-1, 3)
        return new_points

    def interpolate_color(self, mobject1, mobject2, alpha):
        attrs = [
            "fill_color",
            "stroke_color",
            # "opacity", # TODO: This probably doesn't exist anymore because opacity is now moved into the colors
            "reflectiveness",
            "shadow",
            "gloss",
            "stroke_width",
            # TODO: eventually add these attributes to OpenGLVMobject
            # "background_stroke_width",
            # "sheen_direction",
            # "sheen_factor",
        ]

        def interp(obj1, obj2, alpha):
            result = None
            if isinstance(obj1, ManimColor) or isinstance(obj2, ManimColor):
                result = obj1.interpolate(obj2, alpha)
            else:
                result = interpolate(obj1, obj2, alpha)
            return result

        for attr in attrs:
            if alpha == 1.0:
                setattr(self, attr, getattr(mobject2, attr))
                continue

            attr1 = getattr(mobject1, attr)
            attr2 = getattr(mobject2, attr)
            if isinstance(attr1, list) or isinstance(attr2, list):
                result = [
                    interp(elem1, elem2, alpha) for elem1, elem2 in zip(attr1, attr2)
                ]
            else:
                result = interp(attr1, attr2, alpha)
            setattr(self, attr, result)

    # TODO: compare to 3b1b/manim again check if something changed so we don't need the cairo interpolation anymore
    def pointwise_become_partial(
        self, vmobject: OpenGLVMobject, a: float, b: float, remap: bool = True
    ) -> Self:
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

    def get_subcurve(self, a: float, b: float) -> Self:
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

    def apply_function(self, function, make_smooth=False, **kwargs):
        super().apply_function(function, **kwargs)
        if self.make_smooth_after_applying_functions or make_smooth:
            self.make_approximately_smooth()
        return self

    def apply_points_function(self, *args, **kwargs):
        super().apply_points_function(*args, **kwargs)
        return self

    def flip(self, *args, **kwargs):
        super().flip(*args, **kwargs)
        return self


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

    def __init__(self, *vmobjects: OpenGLVMobject, **kwargs: Any):
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

    def set_z(self, z: float) -> Self:
        self.points[..., -1] = z
        return self

    @deprecated(
        since="0.20.0",
        until="0.21.0",
        message="OpenGL has no concept of z_index. Use set_z instead",
    )
    def set_z_index(self, z: float) -> Self:
        return self.set_z(z)

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
        self.submobjects[key] = value  # type: ignore
        self.note_changed_family()


class OpenGLVectorizedPoint(OpenGLPoint, OpenGLVMobject):
    def __init__(
        self,
        location=ORIGIN,
        color=BLACK,
        fill_opacity=0,
        stroke_width=0,
        **kwargs,
    ):
        OpenGLPoint.__init__(self, location, **kwargs)
        OpenGLVMobject.__init__(
            self,
            color=color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs,
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
        super().__init__(**kwargs)
        self.dashed_ratio = dashed_ratio
        self.num_dashes = num_dashes
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


class VHighlight(OpenGLVGroup):
    def __init__(
        self,
        vmobject: OpenGLVMobject,
        n_layers: int = 5,
        color_bounds: tuple[ManimColor, ManimColor] = (GREY_C, GREY_E),
        max_stroke_addition: float = 5.0,
    ):
        outline = vmobject.replicate(n_layers)
        outline.set_fill(opacity=0)
        added_widths = np.linspace(0, max_stroke_addition, n_layers + 1)[1:]
        colors = color_gradient(color_bounds, n_layers)
        for part, added_width, color in zip(reversed(outline), added_widths, colors):
            for sm in part.family_members_with_points():
                sm.set_stroke(
                    width=sm.get_stroke_width() + added_width,
                    color=color,
                )
        super().__init__(*outline)
