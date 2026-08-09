from collections.abc import Callable, Iterable
from typing import Any, Self

import numpy as np

from manim._config import config
from manim.constants import *
from manim.typing import *

Mobject = Any


class Positionable:
    __slots__ = ()
    dim: int

    def align_on_border(
        self,
        direction: Vector3DLike,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        """Direction just needs to be a vector pointing towards side or
        corner in the 2d plane.
        """
        target_point = np.sign(direction) * (
            config["frame_x_radius"],
            config["frame_y_radius"],
            0,
        )
        point_to_align = self.get_critical_point(direction)
        shift_val = target_point - point_to_align - buff * np.array(direction)
        shift_val = shift_val * abs(np.sign(direction))
        self.shift(shift_val)
        return self

    def align_to(
        self,
        mobject_or_point: "Positionable | Point3DLike",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        """Aligns mobject to another :class:`~.Mobject` in a certain direction.

        Examples:
        mob1.align_to(mob2, UP) moves mob1 vertically so that its
        top edge lines ups with mob2's top edge.
        """
        if isinstance(mobject_or_point, Positionable):
            point = mobject_or_point.get_critical_point(direction)
        else:
            point = mobject_or_point

        for dim in range(self.dim):
            if direction[dim] != 0:
                self.set_coord(point[dim], dim, direction)
        return self

    def apply_complex_function(
        self,
        function: Callable[[complex], complex],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Applies a complex function to a :class:`Mobject`.
        The x and y Point3Ds correspond to the real and imaginary parts respectively.

        Example
        -------

        .. manim:: ApplyFuncExample

            class ApplyFuncExample(Scene):
                def construct(self):
                    circ = Circle().scale(1.5)
                    circ_ref = circ.copy()
                    circ.apply_complex_function(
                        lambda x: np.exp(x*1j)
                    )
                    t = ValueTracker(0)
                    circ.add_updater(
                        lambda x: x.become(circ_ref.copy().apply_complex_function(
                            lambda x: np.exp(x+t.get_value()*1j)
                        )).set_color(BLUE)
                    )
                    self.add(circ_ref)
                    self.play(TransformFromCopy(circ_ref, circ))
                    self.play(t.animate.set_value(TAU), run_time=3)
        """

        def R3_func(point: Point3D) -> Point3D:
            x, y, z = point
            xy_complex = function(complex(x, y))
            return np.array([xy_complex.real, xy_complex.imag, z])

        return self.apply_function(
            R3_func, about_point=about_point, about_edge=about_edge
        )

    def apply_function(
        self,
        function: MappingFunction,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if about_point is None and about_edge is None:
            about_point = ORIGIN

        def multi_mapping_function(points: Point3D_Array) -> Point3D_Array:
            result: Point3D_Array = np.apply_along_axis(function, 1, points)
            return result

        self.apply_points_function(
            multi_mapping_function,
            about_point,
            about_edge,
        )
        return self

    def apply_function_to_position(self, function: MappingFunction) -> Self:
        self.move_to(function(self.get_center()))
        return self

    def apply_matrix(
        self,
        matrix: MatrixMN,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        # Default to applying matrix about the origin, not mobjects center
        if about_point is None and about_edge is None:
            about_point = ORIGIN
        full_matrix = np.identity(self.dim)
        matrix = np.array(matrix)
        full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        self.apply_points_function(
            lambda points: np.dot(points, full_matrix.T), about_point, about_edge
        )
        return self

    def apply_over_attr_arrays(self, func: MultiMappingFunction) -> Self:
        for attr in self.get_array_attrs():
            setattr(self, attr, func(getattr(self, attr)))
        return self

    def apply_points_function(
        self,
        func: MultiMappingFunction,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        works_on_bounding_box: bool = False,
    ) -> Self:
        raise NotImplementedError

    def center(self) -> Self:
        """Moves the center of the mobject to the center of the scene.

        Returns
        -------
        :class:`.Mobject`
            The centered mobject.
        """
        self.shift(-self.get_center())
        return self

    @property
    def depth(self) -> float:
        """The depth of the mobject.

        Returns
        -------
        :class:`float`

        See also
        --------
        :meth:`length_over_dim`

        """
        # Get the length across the Z dimension
        return self.length_over_dim(2)

    @depth.setter
    def depth(self, value: float) -> None:
        self.scale_to_fit_depth(value)

    def flip(
        self,
        axis: Vector3DLike = UP,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Flips/Mirrors an mobject about its center.

        Examples
        --------

        .. manim:: FlipExample
            :save_last_frame:

            class FlipExample(Scene):
                def construct(self):
                    s= Line(LEFT, RIGHT+UP).shift(4*LEFT)
                    self.add(s)
                    s2= s.copy().flip()
                    self.add(s2)

        """
        return self.rotate(
            TAU / 2, axis, about_point=about_point, about_edge=about_edge
        )

    def get_array_attrs(self) -> Iterable[str]:
        return []

    def get_bottom(self) -> Point3D:
        """Get bottom Point3Ds of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(DOWN)

    def get_boundary_point(self, direction: Vector3DLike) -> Point3D:
        raise NotImplementedError

    def get_bounding_box(self) -> Point3D_Array:
        raise NotImplementedError

    def get_center(self) -> Point3D:
        raise NotImplementedError

    def get_center_of_mass(self) -> Point3D:
        raise NotImplementedError

    def get_coord(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        raise NotImplementedError

    def get_corner(self, direction: Vector3DLike) -> Point3D:
        """Get corner Point3Ds for certain direction."""
        return self.get_critical_point(direction)

    def get_critical_point(self, direction: Vector3DLike) -> Point3D:
        raise NotImplementedError

    def get_depth(self) -> float:
        """Returns the depth of the mobject."""
        return self.length_over_dim(2)

    def get_edge_center(self, direction: Vector3DLike) -> Point3D:
        """Get edge Point3Ds for certain direction."""
        return self.get_critical_point(direction)

    def get_extremum_along_dim(
        self,
        points: Point3DLike_Array | None = None,
        dim: int = 0,
        key: int = 0,
    ) -> float:
        raise NotImplementedError

    def get_height(self) -> float:
        """Returns the height of the mobject."""
        return self.length_over_dim(1)

    def get_left(self) -> Point3D:
        """Get left Point3Ds of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(LEFT)

    def get_midpoint(self) -> Point3D:
        """Get Point3Ds of the middle of the path that forms the  :class:`~.Mobject`.

        Examples
        --------

        .. manim:: AngleMidPoint
            :save_last_frame:

            class AngleMidPoint(Scene):
                def construct(self):
                    line1 = Line(ORIGIN, 2*RIGHT)
                    line2 = Line(ORIGIN, 2*RIGHT).rotate_about_origin(80*DEGREES)

                    a = Angle(line1, line2, radius=1.5, other_angle=False)
                    d = Dot(a.get_midpoint()).set_color(RED)

                    self.add(line1, line2, a, d)
                    self.wait()

        """
        return self.point_from_proportion(0.5)

    def get_nadir(self) -> Point3D:
        """Get nadir (opposite the zenith) Point3Ds of a box bounding a 3D :class:`~.Mobject`."""
        return self.get_edge_center(IN)

    def get_right(self) -> Point3D:
        """Get right Point3Ds of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(RIGHT)

    def get_top(self) -> Point3D:
        """Get top Point3Ds of a box bounding the :class:`~.Mobject`"""
        return self.get_edge_center(UP)

    def get_width(self) -> float:
        """Returns the width of the mobject."""
        return self.length_over_dim(0)

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns x Point3D of the center of the :class:`~.Mobject` as ``float``"""
        return self.get_coord(0, direction)

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns y Point3D of the center of the :class:`~.Mobject` as ``float``"""
        return self.get_coord(1, direction)

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns z Point3D of the center of the :class:`~.Mobject` as ``float``"""
        return self.get_coord(2, direction)

    def get_zenith(self) -> Point3D:
        """Get zenith Point3Ds of a box bounding a 3D :class:`~.Mobject`."""
        return self.get_edge_center(OUT)

    @property
    def height(self) -> float:
        """The height of the mobject.

        Returns
        -------
        :class:`float`

        Examples
        --------
        .. manim:: HeightExample

            class HeightExample(Scene):
                def construct(self):
                    decimal = DecimalNumber().to_edge(UP)
                    rect = Rectangle(color=BLUE)
                    rect_copy = rect.copy().set_stroke(GRAY, opacity=0.5)

                    decimal.add_updater(lambda d: d.set_value(rect.height))

                    self.add(rect_copy, rect, decimal)
                    self.play(rect.animate.set(height=5))
                    self.wait()

        See also
        --------
        :meth:`length_over_dim`

        """
        # Get the length across the Y dimension
        return self.length_over_dim(1)

    @height.setter
    def height(self, value: float) -> None:
        self.scale_to_fit_height(value)

    def is_off_screen(self) -> bool:
        if self.get_left()[0] > config["frame_x_radius"]:
            return True
        if self.get_right()[0] < -config["frame_x_radius"]:
            return True
        if self.get_bottom()[1] > config["frame_y_radius"]:
            return True
        rv: bool = self.get_top()[1] < -config["frame_y_radius"]
        return rv

    def is_point_touching(
        self, point: Point3DLike, buff: float = MED_SMALL_BUFF
    ) -> bool:
        bb = self.get_bounding_box()
        mins = bb[0] - buff
        maxs = bb[2] + buff
        rv: bool = (point >= mins).all() and (point <= maxs).all()
        return rv

    def length_over_dim(self, dim: int) -> float:
        raise NotImplementedError

    def match_coord(
        self, mobject: Mobject, dim: int, direction: Vector3DLike = ORIGIN
    ) -> Self:
        """Match the Point3Ds with the Point3Ds of another :class:`~.Mobject`."""
        return self.set_coord(
            mobject.get_coord(dim, direction),
            dim=dim,
            direction=direction,
        )

    def match_depth(self, mobject: Mobject, **kwargs: Any) -> Self:
        """Match the depth with the depth of another :class:`~.Mobject`."""
        return self.match_dim_size(mobject, 2, **kwargs)

    def match_dim_size(self, mobject: Mobject, dim: int, **kwargs: Any) -> Self:
        """Match the specified dimension with the dimension of another :class:`~.Mobject`."""
        return self.rescale_to_fit(mobject.length_over_dim(dim), dim, **kwargs)

    def match_height(self, mobject: Mobject, **kwargs: Any) -> Self:
        """Match the height with the height of another :class:`~.Mobject`."""
        return self.match_dim_size(mobject, 1, **kwargs)

    def match_width(self, mobject: Mobject, **kwargs: Any) -> Self:
        """Match the width with the width of another :class:`~.Mobject`."""
        return self.match_dim_size(mobject, 0, **kwargs)

    def match_x(self, mobject: Mobject, direction: Vector3DLike = ORIGIN) -> Self:
        """Match x coord. to the x coord. of another :class:`~.Mobject`."""
        return self.match_coord(mobject, 0, direction)

    def match_y(self, mobject: Mobject, direction: Vector3DLike = ORIGIN) -> Self:
        """Match y coord. to the x coord. of another :class:`~.Mobject`."""
        return self.match_coord(mobject, 1, direction)

    def match_z(self, mobject: Mobject, direction: Vector3DLike = ORIGIN) -> Self:
        """Match z coord. to the x coord. of another :class:`~.Mobject`."""
        return self.match_coord(mobject, 2, direction)

    def move_to(
        self,
        point_or_mobject: Point3DLike | Mobject,
        aligned_edge: Vector3DLike = ORIGIN,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        """Move center of the :class:`~.Mobject` to certain Point3D."""
        if isinstance(point_or_mobject, Positionable):
            target = point_or_mobject.get_critical_point(aligned_edge)
        else:
            target = point_or_mobject
        point_to_align = self.get_critical_point(aligned_edge)
        self.shift((target - point_to_align) * coor_mask)
        return self

    def next_to(
        self,
        mobject_or_point: Mobject | Point3DLike,
        direction: Vector3DLike = RIGHT,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        aligned_edge: Vector3DLike = ORIGIN,
        submobject_to_align: Mobject | None = None,
        index_of_submobject_to_align: int | None = None,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        raise NotImplementedError

    def point_from_proportion(self, alpha: float) -> Point3D:
        raise NotImplementedError("Please override in a child class.")

    def pose_at_angle(self, **kwargs: Any) -> Self:
        self.rotate(TAU / 14, RIGHT + UP, **kwargs)
        return self

    def proportion_from_point(self, point: Point3DLike) -> float:
        raise NotImplementedError("Please override in a child class.")

    def reduce_across_dimension(
        self,
        reduce_func: Callable[[Iterable[float]], float],
        dim: int,
    ) -> float | None:
        raise NotImplementedError

    def rescale_to_fit(
        self,
        length: float,
        dim: int,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        old_length = self.length_over_dim(dim)
        if old_length == 0:
            return self
        if stretch:
            self.stretch(length / old_length, dim, **kwargs)
        else:
            self.scale(length / old_length, **kwargs)
        return self

    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Rotates the :class:`~.Mobject` around a specified axis and point.

        Parameters
        ----------
        angle
            The angle of rotation in radians. Predefined constants such as ``DEGREES``
            can also be used to specify the angle in degrees.
        axis
            The rotation axis (see :class:`~.Rotating` for more).
        about_point
            The point about which the mobject rotates. If ``None``, rotation occurs around
            the center of the mobject.
        about_edge
            The edge about which to apply the scaling.

        Returns
        -------
        :class:`Mobject`
            ``self`` (for method chaining)


        .. note::
            To animate a rotation, use :class:`~.Rotating` or :class:`~.Rotate`
            instead of ``.animate.rotate(...)``.
            The ``.animate.rotate(...)`` syntax only applies a transformation
            from the initial state to the final rotated state
            (interpolation between the two states), without showing proper rotational motion
            based on the angle (from 0 to the given angle).

        Examples
        --------

        .. manim:: RotateMethodExample
            :save_last_frame:

            class RotateMethodExample(Scene):
                def construct(self):
                    circle = Circle(radius=1, color=BLUE)
                    line = Line(start=ORIGIN, end=RIGHT)
                    arrow1 = Arrow(start=ORIGIN, end=RIGHT, buff=0, color=GOLD)
                    group1 = VGroup(circle, line, arrow1)

                    group2 = group1.copy()
                    arrow2 = group2[2]
                    arrow2.rotate(angle=PI / 4, about_point=arrow2.get_start())

                    group3 = group1.copy()
                    arrow3 = group3[2]
                    arrow3.rotate(angle=120 * DEGREES, about_point=arrow3.get_start())

                    self.add(VGroup(group1, group2, group3).arrange(RIGHT, buff=1))

        See also
        --------
        :class:`~.Rotating`, :class:`~.Rotate`, :attr:`~.Mobject.animate`, :meth:`apply_points_function_about_point`

        """
        raise NotImplementedError

    def rotate_about_origin(self, angle: float, axis: Vector3DLike = OUT) -> Self:
        """Rotates the :class:`~.Mobject` about the ORIGIN, which is at [0,0,0]."""
        return self.rotate(angle, axis, about_point=ORIGIN)

    def scale(
        self,
        scale_factor: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        works_on_bounding_box: bool = False,
        **kwargs: Any,
    ) -> Self:
        r"""Scale the size by a factor.

        Default behavior is to scale about the center of the mobject.

        Parameters
        ----------
        scale_factor
            The scaling factor :math:`\alpha`. If :math:`0 < |\alpha| < 1`, the mobject
            will shrink, and for :math:`|\alpha| > 1` it will grow. Furthermore,
            if :math:`\alpha < 0`, the mobject is also flipped.
        about_point
            The point about which to apply the scaling.
        about_edge
            The edge about which to apply the scaling.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------

        .. manim:: MobjectScaleExample
            :save_last_frame:

            class MobjectScaleExample(Scene):
                def construct(self):
                    f1 = Text("F")
                    f2 = Text("F").scale(2)
                    f3 = Text("F").scale(0.5)
                    f4 = Text("F").scale(-1)

                    vgroup = VGroup(f1, f2, f3, f4).arrange(6 * RIGHT)
                    self.add(vgroup)

        See also
        --------
        :meth:`move_to`

        """
        self.apply_points_function(
            lambda points: scale_factor * points,
            about_point=about_point,
            about_edge=about_edge,
            works_on_bounding_box=works_on_bounding_box,
        )
        return self

    def scale_to_fit_depth(
        self,
        depth: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Scales the :class:`~.Mobject` to fit a depth while keeping width/height proportional."""
        return self.rescale_to_fit(depth, 2, stretch=stretch, **kwargs)

    def scale_to_fit_height(
        self,
        height: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Scales the :class:`~.Mobject` to fit a height while keeping width/depth proportional.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.width
            np.float64(2.0)
            >>> sq.scale_to_fit_height(5)
            Square
            >>> sq.height
            np.float64(5.0)
            >>> sq.width
            np.float64(5.0)
        """
        return self.rescale_to_fit(height, 1, stretch=stretch, **kwargs)

    def scale_to_fit_width(
        self,
        width: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Scales the :class:`~.Mobject` to fit a width while keeping height/depth proportional.

        Returns
        -------
        :class:`Mobject`
            ``self``

        Examples
        --------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.height
            np.float64(2.0)
            >>> sq.scale_to_fit_width(5)
            Square
            >>> sq.width
            np.float64(5.0)
            >>> sq.height
            np.float64(5.0)
        """
        return self.rescale_to_fit(width, 0, stretch=stretch, **kwargs)

    def set_coord(
        self,
        value: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        raise NotImplementedError

    def set_depth(
        self,
        depth: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def set_height(
        self,
        height: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def set_width(
        self,
        width: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def set_x(self, x: float, direction: Vector3DLike = ORIGIN) -> Self:
        raise NotImplementedError

    def set_y(self, y: float, direction: Vector3DLike = ORIGIN) -> Self:
        raise NotImplementedError

    def set_z(self, z: float, direction: Vector3DLike = ORIGIN) -> Self:
        raise NotImplementedError

    def shift(self, *vectors: Vector3DLike) -> Self:
        raise NotImplementedError

    def shift_onto_screen(self, **kwargs: Any) -> Self:
        raise NotImplementedError

    def stretch(
        self,
        factor: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        raise NotImplementedError

    def stretch_about_point(self, factor: float, dim: int, point: Point3DLike) -> Self:
        raise NotImplementedError

    def stretch_to_fit_depth(self, depth: float, **kwargs: Any) -> Self:
        raise NotImplementedError

    def stretch_to_fit_height(self, height: float, **kwargs: Any) -> Self:
        raise NotImplementedError

    def stretch_to_fit_width(self, width: float, **kwargs: Any) -> Self:
        raise NotImplementedError

    def to_corner(
        self,
        corner: Vector3DLike = DL,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        raise NotImplementedError

    def to_edge(
        self,
        edge: Vector3DLike = LEFT,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        raise NotImplementedError

    @property
    def width(self) -> float:
        raise NotImplementedError

    @width.setter
    def width(self, value: float) -> None:
        raise NotImplementedError
