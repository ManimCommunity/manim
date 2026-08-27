import operator as op
from collections.abc import Callable, Iterable
from functools import reduce
from typing import Any, Self

import numpy as np

from manim._config import config
from manim.constants import (
    DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    DL,
    DOWN,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    RIGHT,
    TAU,
    UP,
)
from manim.typing import (
    MatrixMN,
    Point3D,
    Point3D_Array,
    Point3DLike,
    Point3DLike_Array,
    Vector3DLike,
)
from manim.utils.space_ops import rotation_matrix


class Positionable:
    """A positionable object."""

    ### FUNDAMENTALS ###
    points: Point3D_Array = np.array([])

    def get_all_points(self) -> Point3D_Array:
        """Returns all points.

        Returns
        -------
        Point3D_Array
            All points.

        See also
        --------
        :meth:`set_points`
        """
        result = self.points
        for mob in self.get_family():
            if mob is self:
                continue
            if len(mob.points) > 0:
                result = np.append(result, mob.points, axis=0)
        return result

    def set_points(
        self,
        points: "Point3DLike_Array | Positionable",
    ) -> Self:
        """Sets the points.

        When another object is passed, the points of per family member is matched.

        Parameters
        ----------
        points : Point3DLike_Array | Positionable
            The points.

        Returns
        -------
        Self
            The object itself.

        Examples
        --------
        .. manim:: MatchPointsScene

            class MatchPointsScene(Scene):
                def construct(self):
                    circ = Circle(fill_color=RED, fill_opacity=0.8)
                    square = Square(fill_color=BLUE, fill_opacity=0.2)
                    self.add(circ)
                    self.wait(0.5)
                    self.play(circ.animate.set_points(square))
                    self.wait(0.5)

        See also
        --------
        :meth:`get_points`
        """
        if isinstance(points, Positionable):
            for mob1, mob2 in zip(self.get_family(), points.get_family(), strict=False):
                mob1.set_points(mob2.points.copy())
        else:
            self.points = np.asarray(points)
        return self

    def get_points_defining_boundary(self) -> Point3D_Array:
        """Returns all points defining the boundary.

        Returns
        -------
        Point3D_Array
            The points defining the boundary.

        See also
        --------
        :meth:`get_points`
        """
        return self.get_all_points()

    ### APPLYING FUNCTIONS ###

    def get_family(self) -> Iterable["Positionable"]:
        """Returns all family members recursively."""
        yield self

    def apply_to_family(
        self,
        function: Callable[["Positionable"], Any],
        *,
        only_with_points: bool = True,
    ) -> Self:
        """Applies a function to every family member.

        Parameters
        ----------
        function : Callable[[Positionable], Any]
            The function to apply.
        only_with_points : bool, optional
            Whether to apply the function only to members with points., by default True

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_family`
        """
        for mob in self.get_family():
            if only_with_points and len(mob.points) == 0:
                continue
            function(mob)
        return self

    def apply_array_function(
        self,
        function: Callable[[Point3D_Array], Point3D_Array],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Applies a function to the points array.

        Parameters
        ----------
        function : Callable[[Point3D_Array], Point3D_Array]
            The function to apply.
        about_point : Point3DLike | None, optional
            The point about which to apply the function., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to apply the function., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`apply_to_family`
        """
        if about_point is None:
            if about_edge is None:
                about_edge = ORIGIN
            about_point = self.get_position(direction=about_edge)

        # TODO: Is this necessary?
        # Make a copy to prevent mutation of the original array if about_point is a view
        about_point = np.array(about_point, copy=True)

        if (about_point == ORIGIN).all():

            def apply(mob: Positionable) -> None:
                mob.points = function(mob.points)
        else:

            def apply(mob: Positionable) -> None:
                mob.points -= about_point
                mob.points = function(mob.points)
                mob.points += about_point

        return self.apply_to_family(function=apply)

    def apply_function(
        self,
        function: Callable[[Point3D], Point3D],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Applies a function to every point.

        Parameters
        ----------
        function : Callable[[Point3D], Point3D]
            The function to apply.
        about_point : Point3DLike | None, optional
            The point about which to apply the function., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to apply the function., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`apply_array_function`
        """
        if about_point is None and about_edge is None:
            about_point = ORIGIN

        def apply(points: Point3D_Array) -> Point3D_Array:
            return np.apply_along_axis(func1d=function, axis=1, arr=points)

        return self.apply_array_function(
            function=apply,
            about_point=about_point,
            about_edge=about_edge,
        )

    def apply_complex_function(
        self,
        function: Callable[[complex], complex],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Applies a complex function to every point.

        Parameters
        ----------
        function : Callable[[complex], complex]
            The function to apply.
        about_point : Point3DLike | None, optional
            The point about which to apply the function., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to apply the function., by default None

        Returns
        -------
        Self
            The object itself.

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

        See also
        --------
        :meth:`apply_function`
        """

        def apply(point: Point3D) -> Point3D:
            x, y, z = point
            xy_complex = function(complex(x, y))
            return np.array([xy_complex.real, xy_complex.imag, z])

        return self.apply_function(
            function=apply,
            about_point=about_point,
            about_edge=about_edge,
        )

    ### TRANSFORMATIONS ###

    def apply_matrix(
        self,
        matrix: MatrixMN,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Applies a matrix to every point.

        Parameters
        ----------
        matrix : MatrixMN
            The matrix to apply.
        about_point : Point3DLike | None, optional
            The point about which to apply the matrix., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to apply the matrix., by default None

        Returns
        -------
        Self
            The object itself.
        """
        if about_point is None and about_edge is None:
            about_point = ORIGIN
        matrix = np.asarray(matrix)
        full_matrix = np.identity(3)
        full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix

        return self.apply_array_function(
            function=lambda points: points.dot(full_matrix.T, out=points),
            about_point=about_point,
            about_edge=about_edge,
        )

    def translate(self, vector: Vector3DLike) -> Self:
        """Applies a translation.

        Parameters
        ----------
        vector : Vector3DLike
            The vector.

        Returns
        -------
        Self
            The object itself.
        """
        return self.apply_to_family(function=lambda mob: mob.points.__iadd__(vector))

    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Applies a rotation.

        Parameters
        ----------
        angle : float
            The angle.
        axis : Vector3DLike, optional
            The axis about which to apply the rotation., by default OUT
        about_point : Point3DLike | None, optional
            The point about which to apply the rotation., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to apply the rotation., by default None

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

        Returns
        -------
        Self
            The object itself.
        """
        if about_point is None and about_edge is None:
            about_edge = ORIGIN
        return self.apply_matrix(
            matrix=rotation_matrix(angle, axis),
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale(
        self,
        factor: float,
        scale_stroke: bool = False,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Applies a uniform scaling.

        Parameters
        ----------
        factor : float
            The factor.
        scale_stroke : bool, optional
            Whether to scale the stroke width., by default False
        about_point : Point3DLike | None, optional
            The point about which to apply the scaling., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to apply the scaling., by default None

        Returns
        -------
        Self
            The object itself.

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
        :meth:`stretch`
        """
        return self.apply_array_function(
            function=lambda points: points.__imul__(factor),
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch(
        self,
        factor: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Applies a non-uniform scaling.

        Parameters
        ----------
        factor : float
            The factor.
        dim : int
            The dimension to scale.
        about_point : Point3DLike | None, optional
            The point about which to apply the stretching., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to apply the stretching., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`scale`
        """

        def function(points: Point3D_Array) -> Point3D_Array:
            points[:, dim] *= factor
            return points

        return self.apply_array_function(
            function=function,
            about_point=about_point,
            about_edge=about_edge,
        )

    ### GENERAL ###

    def get_position(
        self,
        direction: Vector3DLike = ORIGIN,
    ) -> Point3D:
        """The position.

        Parameters
        ----------
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        Point3D
            The position.

        See also
        --------
        :meth:`set_position`
        """
        points = self.get_points_defining_boundary()
        return np.array(
            [
                self._get_extremum_along_dim(
                    points=points,
                    dim=dim,
                    key=key,
                )
                for dim, key in enumerate(direction)
            ]
        )

    def set_position(
        self,
        point: "Point3DLike | Positionable",
        *,
        aligned_edge: Vector3DLike = ORIGIN,
    ) -> Self:
        """Sets the position.

        Parameters
        ----------
        point : Point3DLike | Positionable
            The point.
        aligned_edge : Vector3DLike, optional
            Which edge to position., by default ORIGIN

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_position`
        """
        if isinstance(point, Positionable):
            point = point.get_position(direction=aligned_edge)
        current = self.get_position(direction=aligned_edge)
        vector = point - current
        return self.translate(vector=vector)

    def get_center(self) -> Point3D:
        """Returns the center position.

        Returns
        -------
        Point3D
            The center position.

        See also
        --------
        :meth:`set_center`, :meth:`get_position`
        """
        return self.get_position(direction=ORIGIN)

    def set_center(
        self,
        center: "Point3DLike | Positionable",
    ) -> Self:
        """Sets the center position.

        Parameters
        ----------
        center : Point3DLike | Positionable
            The center position.

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_center`, :meth:`set_position`
        """
        return self.set_position(point=center, aligned_edge=ORIGIN)

    def get_left(self) -> Point3D:
        """Returns the left position.

        Returns
        -------
        Point3D
            The left position.

        See also
        --------
        :meth:`set_left`, :meth:`get_position`
        """
        return self.get_position(direction=LEFT)

    def set_left(
        self,
        left: "Point3DLike | Positionable",
    ) -> Self:
        """Sets the left position.

        Parameters
        ----------
        left : Point3DLike | Positionable
            The left position.

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_left`, :meth:`set_position`
        """
        return self.set_position(point=left, aligned_edge=LEFT)

    def get_right(self) -> Point3D:
        """Returns the right position.

        Returns
        -------
        Point3D
            The right position.

        See also
        --------
        :meth:`set_right`, :meth:`get_position`
        """
        return self.get_position(direction=RIGHT)

    def set_right(
        self,
        right: "Point3DLike | Positionable",
    ) -> Self:
        """Sets the right position.

        Parameters
        ----------
        right : Point3DLike | Positionable
            The right position.

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_right`, :meth:`set_position`
        """
        return self.set_position(point=right, aligned_edge=RIGHT)

    def get_bottom(self) -> Point3D:
        """Returns the bottom position.

        Returns
        -------
        Point3D
            The bottom position.

        See also
        --------
        :meth:`set_bottom`, :meth:`get_position`
        """
        return self.get_position(direction=DOWN)

    def set_bottom(
        self,
        bottom: "Point3DLike | Positionable",
    ) -> Self:
        """Sets the bottom position.

        Parameters
        ----------
        bottom : Point3DLike | Positionable
            The bottom position.

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_bottom`, :meth:`set_position`
        """
        return self.set_position(point=bottom, aligned_edge=DOWN)

    def get_top(self) -> Point3D:
        """Returns the top position.

        Returns
        -------
        Point3D
            The top position.

        See also
        --------
        :meth:`set_top`, :meth:`get_position`
        """
        return self.get_position(direction=UP)

    def set_top(
        self,
        top: "Point3DLike | Positionable",
    ) -> Self:
        """Sets the top position.

        Parameters
        ----------
        top : Point3DLike | Positionable
            The top position.

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_top`, :meth:`set_position`
        """
        return self.set_position(point=top, aligned_edge=UP)

    def get_nadir(self) -> Point3D:
        """Returns the nadir position.

        Returns
        -------
        Point3D
            The  nadir position.

        See also
        --------
        :meth:`set_nadir`, :meth:`get_position`
        """
        return self.get_position(direction=IN)

    def set_nadir(
        self,
        nadir: "Point3DLike | Positionable",
    ) -> Self:
        """Sets the nadir position.

        Parameters
        ----------
        nadir : Point3DLike | Positionable
            The nadir position.

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_nadir`, :meth:`set_position`
        """
        return self.set_position(point=nadir, aligned_edge=IN)

    def get_zenith(self) -> Point3D:
        """Returns the zenith position.

        Returns
        -------
        Point3D
            The zenith position.

        See also
        --------
        :meth:`set_zenith`, :meth:`get_position`
        """
        return self.get_position(direction=OUT)

    def set_zenith(
        self,
        zenith: "Point3DLike | Positionable",
    ) -> Self:
        """Sets the zenith position.

        Parameters
        ----------
        zenith : Point3DLike | Positionable
            The zenith position.

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_zenith`, :meth:`set_position`
        """
        return self.set_position(point=zenith, aligned_edge=OUT)

    def get_coordinate(
        self,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> float:
        """Returns the coordinate of a dimension.

        Parameters
        ----------
        dim : int
            The dimension.
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        float
            The coordinate.

        See also
        --------
        :meth:`set_coordinate`
        """
        return self._get_extremum_along_dim(
            points=self.get_points_defining_boundary(),
            dim=dim,
            key=np.sign(direction[dim]),
        )

    def set_coordinate(
        self,
        value: "float | Positionable",
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        """Sets the coordinate of a dimension.

        Parameters
        ----------
        value : float | Positionable
            The coordinate.
        dim : int
            The dimension.
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_coordinate`
        """
        if isinstance(value, Positionable):
            value = value.get_coordinate(dim=dim, direction=direction)
        current = self.get_coordinate(dim=dim, direction=direction)
        vector = np.zeros(3)
        vector[dim] = value - current
        return self.translate(vector=vector)

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the x coordinate.

        Parameters
        ----------
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        float
            The x coordinate.

        See also
        --------
        :meth:`set_x`, :meth:`get_coordinate`
        """
        return self.get_coordinate(dim=0, direction=direction)

    def set_x(self, x: float, direction: Vector3DLike = ORIGIN) -> Self:
        """Sets the x coordinate.

        Parameters
        ----------
        x : float
            The x coordinate.
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_x`, :meth:`set_coordinate`
        """
        return self.set_coordinate(value=x, dim=0, direction=direction)

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the y coordinate.

        Parameters
        ----------
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        float
            The y coordinate.

        See also
        --------
        :meth:`set_y`, :meth:`get_coordinate`
        """
        return self.get_coordinate(dim=1, direction=direction)

    def set_y(self, y: float, direction: Vector3DLike = ORIGIN) -> Self:
        """Sets the y coordinate.

        Parameters
        ----------
        y : float
            The y coordinate.
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_y`, :meth:`set_coordinate`
        """
        return self.set_coordinate(value=y, dim=1, direction=direction)

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the z coordinate.

        Parameters
        ----------
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        float
            The z coordinate.

        See also
        --------
        :meth:`set_z`, :meth:`get_coordinate`
        """
        return self.get_coordinate(dim=2, direction=direction)

    def set_z(self, z: float, direction: Vector3DLike = ORIGIN) -> Self:
        """Sets the z coordinate.

        Parameters
        ----------
        z : float
            The z coordinate.
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_z`, :meth:`get_coordinate`
        """
        return self.set_coordinate(value=z, dim=2, direction=direction)

    def get_dim_size(self, dim: int) -> float:
        """Returns the size of a dimension.

        Parameters
        ----------
        dim : int
            The dimension.

        Returns
        -------
        float
            The size of the dimension.

        See also
        --------
        :meth:`set_dim_size`
        """
        # TODO: Changing this to `get_boundary_points` breaks the `test_img_and_svg.py`` tests
        points = self.get_all_points()
        if len(points) == 0:
            return 0
        return np.ptp(points[:, dim])  # type: ignore[no-any-return]

    def set_dim_size(
        self,
        size: "float | Positionable",
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Sets the size of a dimension.

        Parameters
        ----------
        size : float | Positionable
            The size.
        dim : int
            The dimension.
        stretch : bool, optional
            Whether to use non-uniform or uniform scaling., by default False
        about_point : Point3DLike | None, optional
            The point about which the scaling is applied., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which the scaling is applied., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_dim_size`, :meth:`scale`, :meth:`stretch`
        """
        if isinstance(size, Positionable):
            size = size.get_dim_size(dim=dim)

        current_size = self.get_dim_size(dim=dim)
        if current_size == 0:
            return self

        factor = size / current_size
        if stretch:
            return self.stretch(
                factor=factor,
                dim=dim,
                about_point=about_point,
                about_edge=about_edge,
            )
        else:
            return self.scale(
                factor=factor,
                about_point=about_point,
                about_edge=about_edge,
            )

    def get_width(self) -> float:
        """Returns the width.

        Returns
        -------
        float
            The width.

        See also
        --------
        :meth:`set_width`, :meth:`get_dim_size`
        """
        return self.get_dim_size(dim=0)

    def set_width(
        self,
        width: "float | Positionable",
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Sets the width.

        Parameters
        ----------
        width : float | Positionable
            The width.
        stretch : bool, optional
            Whether to use non-uniform or uniform scaling., by default False
        about_point : Point3DLike | None, optional
            The point about which the scaling is applied., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which the scaling is applied., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`get_width`, :meth:`set_dim_size`
        """
        return self.set_dim_size(
            size=width,
            dim=0,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    def get_height(self) -> float:
        """Returns the height.

        Returns
        -------
        float
            The height.

        See also
        --------
        :meth:`set_height`, :meth:`get_dim_size`
        """
        return self.get_dim_size(dim=1)

    def set_height(
        self,
        height: "float | Positionable",
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Sets the height.

        Parameters
        ----------
        height : float | Positionable
            The height.
        stretch : bool, optional
            Whether to use non-uniform or uniform scaling., by default False
        about_point : Point3DLike | None, optional
            The point about which the scaling is applied., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which the scaling is applied., by default None

        Returns
        -------
        Self
            The object itself.


        See also
        --------
        :meth:`get_height`, :meth:`set_dim_size`
        """
        return self.set_dim_size(
            size=height,
            dim=1,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    def get_depth(self) -> float:
        """Returns the depth.

        Returns
        -------
        float
            The depth.


        See also
        --------
        :meth:`set_depth`, :meth:`get_dim_size`
        """
        return self.get_dim_size(dim=2)

    def set_depth(
        self,
        depth: "float | Positionable",
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Sets the depth.

        Parameters
        ----------
        depth : float | Positionable
            The depth.
        stretch : bool, optional
            Whether to use non-uniform or uniform scaling., by default False
        about_point : Point3DLike | None, optional
            The point about which the scaling is applied., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which the scaling is applied., by default None

        Returns
        -------
        Self
            The object itself.


        See also
        --------
        :meth:`get_depth`, :meth:`set_dim_size`
        """
        return self.set_dim_size(
            size=depth,
            dim=2,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    ### SPECIALIZED ###

    def align_on_border(
        self,
        direction: Vector3DLike,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        frame: Point3DLike | None = None,
    ) -> Self:
        """Aligns the object on a border.

        Parameters
        ----------
        direction : Vector3DLike
            Which border to align to.
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_EDGE_BUFFER
        frame : Point3DLike | None, optional
            The frame., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`align_to`
        """
        if frame is None:
            frame = (config.frame_x_radius, config.frame_y_radius, 0)
        target = np.sign(direction) * frame - buff * np.asarray(direction)
        return self.align_to(target, direction=direction)

    def align_to(
        self,
        point: "Point3DLike | Positionable",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        """Aligns the object onto a point.

        Parameters
        ----------
        mobject_or_point : Point3DLike | Positionable
            The point.
        direction : Vector3DLike, optional
            TODO, by default ORIGIN

        Returns
        -------
        Self
            The object itself.

        Examples:
        mob1.align_to(mob2, UP) moves mob1 vertically so that its
        top edge lines ups with mob2's top edge.
        """
        if isinstance(point, Positionable):
            point = point.get_position(direction=direction)
        source = self.get_position(direction=direction)
        target = np.where(direction == 0, source, point)
        return self.translate(target - source)

    def get_bounding_box(self) -> tuple[Point3D, Point3D]:
        """Returns the bounding box.

        Returns
        -------
        tuple[Point3D, Point3D]
            The bottom-left and top-right points.
        """
        points = self.get_points_defining_boundary()
        if len(points) == 0:
            return (np.zeros(3), np.zeros(3))
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        return (mins, maxs)

    def get_extremum_along_dim(
        self,
        dim: int = 0,
        key: int = 0,
    ) -> float:
        return self._get_extremum_along_dim(
            self.get_points_defining_boundary(),
            dim=dim,
            key=key,
        )

    def _get_extremum_along_dim(
        self,
        points: Point3D_Array,
        dim: int = 0,
        key: int = 0,
    ) -> float:
        if len(points) == 0:
            return 0
        values = points[:, dim]
        return (  # type: ignore[no-any-return]
            values.min()
            if key < 0
            else (values.min() + values.max()) / 2
            if key == 0
            else values.max()
        )

    def get_center_of_mass(self) -> Point3D:
        """Returns the center of mass.

        Returns
        -------
        Point3D
            The center of mass.
        """
        points = self.get_all_points()
        if len(points) == 0:
            return ORIGIN
        return points.mean(axis=0)

    def get_boundary_point(self, direction: Vector3DLike) -> Point3D:
        """Returns a boundary point.

        Parameters
        ----------
        direction : Vector3DLike
            TODO

        Returns
        -------
        Point3D
            The boundary point.
        """
        points = self.get_points_defining_boundary()
        index = np.argmax(points.dot(direction))
        return points[index]

    def is_off_screen(self) -> bool:
        """Returns whether this is off screen.

        Returns
        -------
        bool
            Is off screen.
        """
        points = self.get_points_defining_boundary()
        return (
            # left is too right
            (
                self._get_extremum_along_dim(points=points, dim=0, key=-1)
                > config.frame_x_radius
            )
            # right is too left
            or (
                self._get_extremum_along_dim(points=points, dim=0, key=1)
                < -config.frame_x_radius
            )
            # bottom is too high
            or (
                self._get_extremum_along_dim(points=points, dim=1, key=-1)
                > config.frame_y_radius
            )
            # top is too low
            or (
                self._get_extremum_along_dim(points=points, dim=1, key=1)
                < -config.frame_y_radius
            )
        )

    def shift_onto_screen(
        self,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        """Shifts onto screen.

        Parameters
        ----------
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_EDGE_BUFFER

        Returns
        -------
        Self
            The object itself.
        """
        # TODO: Simplify/Optimize implementation
        space_lengths = [config.frame_x_radius, config.frame_y_radius]
        for vect in UP, DOWN, LEFT, RIGHT:
            dim = np.argmax(np.abs(vect))
            max_val = space_lengths[dim] - buff
            edge_center = self.get_position(vect)
            if np.dot(edge_center, vect) > max_val:
                self.to_edge(vect, buff=buff)
        return self

    ### ALIASES ###
    get_critical_point = get_position
    get_edge_center = get_position
    get_corner = get_position
    length_over_dim = get_dim_size
    get_coord = get_coordinate
    set_coord = set_coordinate

    def shift(self, *vectors: Vector3DLike) -> Self:
        """_summary_

        Parameters
        ----------
        vectors: *Vector3DLike
            The vectors.

        Returns
        -------
        Self
            The object itself.
        """
        return self.translate(vector=reduce(op.add, vectors))

    def center(self) -> Self:
        """Moves to the ORIGIN.

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`set_center`
        """
        return self.set_center(ORIGIN)

    def flip(
        self,
        axis: Vector3DLike = UP,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Flips.

        Parameters
        ----------
        axis : Vector3DLike, optional
            The axis about which to flip., by default UP
        about_point : Point3DLike | None, optional
            The point about which to flip., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to flip., by default None

        Returns
        -------
        Self
            The object itself.

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

        See also
        --------
        :meth:`rotate`
        """
        return self.rotate(
            TAU / 2,
            axis,
            about_point=about_point,
            about_edge=about_edge,
        )

    def move_to(
        self,
        point_or_mobject: "Point3DLike | Positionable",
        aligned_edge: Vector3DLike = ORIGIN,
    ) -> Self:
        """Moves to a position.

        Parameters
        ----------
        point_or_mobject : Point3DLike | Positionable
            The point.
        aligned_edge : Vector3DLike, optional
            Which edge to position., by default ORIGIN

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`set_position`
        """
        return self.set_position(
            point=point_or_mobject,
            aligned_edge=aligned_edge,
        )

    def pose_at_angle(
        self,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Poses at angle.

        Parameters
        ----------
        about_point : Point3DLike | None, optional
            The point about which to pose., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to pose., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`rotate`
        """
        return self.rotate(
            angle=TAU / 14,
            axis=RIGHT + UP,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit(
        self,
        size: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Scales to fit a size for a dimension.

        Parameters
        ----------
        size : float
            The size.
        dim : int
            The dimension.
        about_point : Point3DLike | None, optional
            The point about which to scale., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to scale., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`set_dim_size`, :meth:`scale`
        """
        return self.set_dim_size(
            size=size,
            dim=dim,
            stretch=False,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit_width(
        self,
        width: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Scales to fit a width.

        Parameters
        ----------
        width : float
            The width.
        about_point : Point3DLike | None, optional
            The point about which scale., by default None
        about_edge : Vector3DLike | None, optional
            The point about which to scale., by default None

        Returns
        -------
        Self
            The object itself.

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

        See also
        --------
        :meth:`scale_to_fit`, :meth:`scale`, :meth:`set_width`
        """
        return self.scale_to_fit(
            size=width,
            dim=0,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit_height(
        self,
        height: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Scales to fit a jeight.

        Parameters
        ----------
        height : float
            The height.
        about_point : Point3DLike | None, optional
            The point about which scale., by default None
        about_edge : Vector3DLike | None, optional
            The point about which to scale., by default None

        Returns
        -------
        Self
            The object itself.

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

        See also
        --------
        :meth:`scale_to_fit`, :meth:`scale`, :meth:`set_height`
        """
        return self.scale_to_fit(
            size=height,
            dim=1,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit_depth(
        self,
        depth: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Scales to fit a depth.

        Parameters
        ----------
        depth : float
            The depth.
        about_point : Point3DLike | None, optional
            The point about which scale., by default None
        about_edge : Vector3DLike | None, optional
            The point about which to scale., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`scale_to_fit`, :meth:`scale`, :meth:`set_depth`
        """
        return self.scale_to_fit(
            size=depth,
            dim=2,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit(
        self,
        size: float,
        dim: int,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Stretches to fit the size of a dimension.

        Parameters
        ----------
        size : float
            The size.
        dim : int
            The dimension.
        about_point : Point3DLike | None, optional
            The point about which to stretch., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to stretch., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`set_dim_size`, :meth:`stretch`
        """
        return self.set_dim_size(
            size=size,
            dim=dim,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_width(
        self,
        width: float,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Stretches to fit a width.

        Parameters
        ----------
        width : float
            The width.
        about_point : Point3DLike | None, optional
            The point about which to stretch., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to stretch., by default None

        Returns
        -------
        Self
            The object itself.

        Examples
        --------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.height
            np.float64(2.0)
            >>> sq.stretch_to_fit_width(5)
            Square
            >>> sq.width
            np.float64(5.0)
            >>> sq.height
            np.float64(2.0)

        See also
        --------
        :meth:`stretch_to_fit`, :meth:`stretch`, :meth:`set_width`
        """
        return self.stretch_to_fit(
            size=width,
            dim=0,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_height(
        self,
        height: float,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Stretches to fit a height.

        Parameters
        ----------
        height : float
            The height.
        about_point : Point3DLike | None, optional
            The point about which to stretch., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to stretch., by default None

        Returns
        -------
        Self
            The object itself.

        Examples
        --------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.width
            np.float64(2.0)
            >>> sq.stretch_to_fit_height(5)
            Square
            >>> sq.height
            np.float64(5.0)
            >>> sq.width
            np.float64(2.0)

        See also
        --------
        :meth:`stretch_to_fit`, :meth:`stretch`, :meth:`set_height`
        """
        return self.stretch_to_fit(
            size=height,
            dim=1,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_depth(
        self,
        depth: float,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Stretches to fit a depth.

        Parameters
        ----------
        depth : float
            The depth.
        about_point : Point3DLike | None, optional
            The point about which to stretch., by default None
        about_edge : Vector3DLike | None, optional
            The edge about which to stretch., by default None

        Returns
        -------
        Self
            The object itself.

        See also
        --------
        :meth:`stretch_to_fit`, :meth:`stretch`, :meth:`set_depth`
        """
        return self.stretch_to_fit(
            size=depth,
            dim=2,
            about_point=about_point,
            about_edge=about_edge,
        )

    def to_corner(
        self,
        corner: Vector3DLike = DL,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        """Aligns to a corner.

        Parameters
        ----------
        corner : Vector3DLike, optional
            The corner., by default DL
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_EDGE_BUFFER

        Returns
        -------
        Self
            The object itself.

        Examples
        --------

        .. manim:: ToCornerExample
            :save_last_frame:

            class ToCornerExample(Scene):
                def construct(self):
                    c = Circle()
                    c.to_corner(UR)
                    t = Tex("To the corner!")
                    t2 = MathTex("x^3").shift(DOWN)
                    self.add(c,t,t2)
                    t.to_corner(DL, buff=0)
                    t2.to_corner(UL, buff=1.5)

        See also
        --------
        :meth:`align_on_border`
        """
        return self.align_on_border(direction=corner, buff=buff)

    def to_edge(
        self,
        edge: Vector3DLike = LEFT,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        """Aligns to an edge.

        Parameters
        ----------
        edge : Vector3DLike, optional
            The edge., by default LEFT
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_EDGE_BUFFER

        Returns
        -------
        Self
            The object itself.

        Examples
        --------

        .. manim:: ToEdgeExample
            :save_last_frame:

            class ToEdgeExample(Scene):
                def construct(self):
                    tex_top = Tex("I am at the top!")
                    tex_top.to_edge(UP)
                    tex_side = Tex("I am moving to the side!")
                    c = Circle().shift(2*DOWN)
                    self.add(tex_top, tex_side, c)
                    tex_side.to_edge(LEFT)
                    c.to_edge(RIGHT, buff=0)

        See also
        --------
        :meth:`align_on_border`
        """
        return self.align_on_border(direction=edge, buff=buff)

    @property
    def width(self) -> float:
        """The width.

        Examples
        --------
        .. manim:: WidthExample

            class WidthExample(Scene):
                def construct(self):
                    decimal = DecimalNumber().to_edge(UP)
                    rect = Rectangle(color=BLUE)
                    rect_copy = rect.copy().set_stroke(GRAY, opacity=0.5)

                    decimal.add_updater(lambda d: d.set_value(rect.width))

                    self.add(rect_copy, rect, decimal)
                    self.play(rect.animate.set(width=7))
                    self.wait()

        See also
        --------
        :meth:`get_width`, :meth:`set_width`
        """
        return self.get_width()

    @width.setter
    def width(self, value: float) -> None:
        self.set_width(width=value)

    @property
    def height(self) -> float:
        """The height.

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
        :meth:`get_height`, :meth:`set_height`
        """
        return self.get_height()

    @height.setter
    def height(self, value: float) -> None:
        self.set_height(height=value)

    @property
    def depth(self) -> float:
        """The width.

        See also
        --------
        :meth:`get_depth`, :meth:`set_depth`
        """
        return self.get_depth()

    @depth.setter
    def depth(self, value: float) -> None:
        self.set_depth(depth=value)

    ### DEPRECATED ###

    apply_points_function_about_point = apply_array_function
    match_points = set_points
    match_coord = set_coordinate
    match_x = set_x
    match_y = set_y
    match_z = set_z
    match_dim_size = set_dim_size
    match_width = set_width
    match_height = set_height
    match_depth = set_depth

    # @deprecated(replacement="move_to(function(self.get_center()))")
    def apply_function_to_position(
        self,
        function: Callable[[Point3D], Point3DLike],
    ) -> Self:
        return self.move_to(function(self.get_center()))

    # @deprecated()
    def reduce_across_dimension(
        self,
        reduce_func: Callable[[Iterable[float]], float],
        dim: int,
    ) -> float | None:
        points = self.get_points_defining_boundary()
        if len(points) == 0:
            return None

        return reduce_func(points[:, dim])

    # @deprecated(replacement="rotate")
    def rotate_about_origin(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
    ) -> Self:
        return self.rotate(
            angle=angle,
            axis=axis,
            about_point=ORIGIN,
        )

    # @deprecated(replacement="set_dim_size")
    def rescale_to_fit(
        self,
        length: "float | Positionable",
        dim: int,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_dim_size(
            size=length,
            dim=dim,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="stretch")
    def stretch_about_point(self, factor: float, dim: int, point: Point3DLike) -> Self:
        return self.stretch(factor=factor, dim=dim, about_point=point)
