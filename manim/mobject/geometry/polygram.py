r"""Mobjects that are simple geometric shapes."""

from __future__ import annotations

__all__ = [
    "Polygram",
    "Polygon",
    "RegularPolygram",
    "RegularPolygon",
    "Star",
    "Triangle",
    "Rectangle",
    "Square",
    "RoundedRectangle",
    "Cutout",
    "ConvexHull",
]


from math import ceil
from typing import TYPE_CHECKING

import numpy as np

from manim.constants import *
from manim.mobject.geometry.arc import ArcBetweenPoints
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.color import BLUE, WHITE, ParsableManimColor
from manim.utils.iterables import adjacent_n_tuples, adjacent_pairs
from manim.utils.qhull import QuickHull
from manim.utils.space_ops import angle_between_vectors, normalize, regular_vertices

if TYPE_CHECKING:
    from typing import Any, Literal

    import numpy.typing as npt
    from typing_extensions import Self

    from manim.typing import (
        ManimFloat,
        Point3D,
        Point3D_Array,
        Point3DLike,
        Point3DLike_Array,
    )
    from manim.utils.color import ParsableManimColor


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

    def __init__(
        self,
        *vertex_groups: Point3DLike_Array,
        color: ParsableManimColor = BLUE,
        **kwargs: Any,
    ):
        super().__init__(color=color, **kwargs)

        for vertices in vertex_groups:
            # The inferred type for *vertices is Any, but it should be
            # Point3D_Array
            first_vertex, *vertices = vertices
            first_vertex = np.array(first_vertex)

            self.start_new_path(first_vertex)
            self.add_points_as_corners(
                [*(np.array(vertex) for vertex in vertices), first_vertex],
            )

    def get_vertices(self) -> Point3D_Array:
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

    def get_vertex_groups(self) -> npt.NDArray[ManimFloat]:
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

    def round_corners(
        self,
        radius: float | list[float] = 0.5,
        evenly_distribute_anchors: bool = False,
        components_per_rounded_corner: int = 2,
    ) -> Self:
        """Rounds off the corners of the :class:`Polygram`.

        Parameters
        ----------
        radius
            The curvature of the corners of the :class:`Polygram`.
        evenly_distribute_anchors
            Break long line segments into proportionally-sized segments.
        components_per_rounded_corner
            The number of points used to represent the rounded corner curve.


        .. seealso::
            :class:`.~RoundedRectangle`

        .. note::
            If `radius` is supplied as a single value, then the same radius
            will be applied to all corners.  If `radius` is a list, then the
            individual values will be applied sequentially, with the first
            corner receiving `radius[0]`, the second corner receiving
            `radius[1]`, etc.  The radius list will be repeated as necessary.

            The `components_per_rounded_corner` value is provided so that the
            fidelity of the rounded corner may be fine-tuned as needed.  2 is
            an appropriate value for most shapes, however a larger value may be
            need if the rounded corner is particularly large.  2 is the minimum
            number allowed, representing the start and end of the curve.  3 will
            result in a start, middle, and end point, meaning 2 curves will be
            generated.

            The option to `evenly_distribute_anchors` is provided so that the
            line segments (the part part of each line remaining after rounding
            off the corners) can be subdivided to a density similar to that of
            the average density of the rounded corners.  This may be desirable
            in situations in which an even distribution of curves is desired
            for use in later transformation animations.  Be aware, though, that
            enabling this option can result in an an object containing
            significantly more points than the original, especially when the
            rounded corner curves are small.

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
        """
        if radius == 0:
            return self

        new_points: list[Point3D] = []

        for vertices in self.get_vertex_groups():
            arcs = []

            # Repeat the radius list as necessary in order to provide a radius
            # for each vertex.
            if isinstance(radius, (int, float)):
                radius_list = [radius] * len(vertices)
            else:
                radius_list = radius * ceil(len(vertices) / len(radius))

            for currentRadius, (v1, v2, v3) in zip(
                radius_list, adjacent_n_tuples(vertices, 3)
            ):
                vect1 = v2 - v1
                vect2 = v3 - v2
                unit_vect1 = normalize(vect1)
                unit_vect2 = normalize(vect2)

                angle = angle_between_vectors(vect1, vect2)
                # Negative radius gives concave curves
                angle *= np.sign(currentRadius)

                # Distance between vertex and start of the arc
                cut_off_length = currentRadius * np.tan(angle / 2)

                # Determines counterclockwise vs. clockwise
                sign = np.sign(np.cross(vect1, vect2)[2])

                arc = ArcBetweenPoints(
                    v2 - unit_vect1 * cut_off_length,
                    v2 + unit_vect2 * cut_off_length,
                    angle=sign * angle,
                    num_components=components_per_rounded_corner,
                )
                arcs.append(arc)

            if evenly_distribute_anchors:
                # Determine the average length of each curve
                nonZeroLengthArcs = [arc for arc in arcs if len(arc.points) > 4]
                if len(nonZeroLengthArcs):
                    totalArcLength = sum(
                        [arc.get_arc_length() for arc in nonZeroLengthArcs]
                    )
                    totalCurveCount = (
                        sum([len(arc.points) for arc in nonZeroLengthArcs]) / 4
                    )
                    averageLengthPerCurve = totalArcLength / totalCurveCount
                else:
                    averageLengthPerCurve = 1

            # To ensure that we loop through starting with last
            arcs = [arcs[-1], *arcs[:-1]]
            from manim.mobject.geometry.line import Line

            for arc1, arc2 in adjacent_pairs(arcs):
                new_points.extend(arc1.points)

                line = Line(arc1.get_end(), arc2.get_start())

                # Make sure anchors are evenly distributed, if necessary
                if evenly_distribute_anchors:
                    line.insert_n_curves(
                        ceil(line.get_length() / averageLengthPerCurve)
                    )

                new_points.extend(line.points)

        self.set_points(np.array(new_points))

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

    def __init__(self, *vertices: Point3DLike, **kwargs: Any) -> None:
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
        start_angle: float | None = None,
        **kwargs: Any,
    ) -> None:
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
        def gen_polygon_vertices(start_angle: float | None) -> tuple[list[Any], float]:
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

    def __init__(self, n: int = 6, **kwargs: Any) -> None:
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
        inner_radius: float | None = None,
        density: int = 2,
        start_angle: float | None = TAU / 4,
        **kwargs: Any,
    ) -> None:
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

        vertices: list[npt.NDArray] = []
        for pair in zip(outer_vertices, inner_vertices):
            vertices.extend(pair)

        super().__init__(*vertices, **kwargs)


class Triangle(RegularPolygon):
    """An equilateral triangle.

    Parameters
    ----------
    kwargs
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(n=3, **kwargs)


class Rectangle(Polygon):
    """A quadrilateral with two sets of parallel sides.

    Parameters
    ----------
    color
        The color of the rectangle.
    height
        The vertical height of the rectangle.
    width
        The horizontal width of the rectangle.
    grid_xstep
        Space between vertical grid lines.
    grid_ystep
        Space between horizontal grid lines.
    mark_paths_closed
        No purpose.
    close_new_points
        No purpose.
    kwargs
        Additional arguments to be passed to :class:`Polygon`

    Examples
    ----------
    .. manim:: RectangleExample
        :save_last_frame:

        class RectangleExample(Scene):
            def construct(self):
                rect1 = Rectangle(width=4.0, height=2.0, grid_xstep=1.0, grid_ystep=0.5)
                rect2 = Rectangle(width=1.0, height=4.0)
                rect3 = Rectangle(width=2.0, height=2.0, grid_xstep=1.0, grid_ystep=1.0)
                rect3.grid_lines.set_stroke(width=1)

                rects = Group(rect1, rect2, rect3).arrange(buff=1)
                self.add(rects)
    """

    def __init__(
        self,
        color: ParsableManimColor = WHITE,
        height: float = 2.0,
        width: float = 4.0,
        grid_xstep: float | None = None,
        grid_ystep: float | None = None,
        mark_paths_closed: bool = True,
        close_new_points: bool = True,
        **kwargs: Any,
    ):
        super().__init__(UR, UL, DL, DR, color=color, **kwargs)
        self.stretch_to_fit_width(width)
        self.stretch_to_fit_height(height)

        v = self.get_vertices()
        self.grid_lines = VGroup()

        if grid_xstep or grid_ystep:
            from manim.mobject.geometry.line import Line

            v = self.get_vertices()

        if grid_xstep:
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
            self.grid_lines.add(grid)

        if grid_ystep:
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
            self.grid_lines.add(grid)

        if self.grid_lines:
            self.add(self.grid_lines)


class Square(Rectangle):
    """A rectangle with equal side lengths.

    Parameters
    ----------
    side_length
        The length of the sides of the square.
    kwargs
        Additional arguments to be passed to :class:`Rectangle`.

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

    def __init__(self, side_length: float = 2.0, **kwargs: Any) -> None:
        super().__init__(height=side_length, width=side_length, **kwargs)

    @property
    def side_length(self) -> float:
        return float(np.linalg.norm(self.get_vertices()[0] - self.get_vertices()[1]))

    @side_length.setter
    def side_length(self, value: float) -> None:
        self.scale(value / self.side_length)


class RoundedRectangle(Rectangle):
    """A rectangle with rounded corners.

    Parameters
    ----------
    corner_radius
        The curvature of the corners of the rectangle.
    kwargs
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

    def __init__(self, corner_radius: float | list[float] = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.corner_radius = corner_radius
        self.round_corners(self.corner_radius)


class Cutout(VMobject, metaclass=ConvertToOpenGL):
    """A shape with smaller cutouts.

    Parameters
    ----------
    main_shape
        The primary shape from which cutouts are made.
    mobjects
        The smaller shapes which are to be cut out of the ``main_shape``.
    kwargs
        Further keyword arguments that are passed to the constructor of
        :class:`~.VMobject`.


    .. warning::
        Technically, this class behaves similar to a symmetric difference: if
        parts of the ``mobjects`` are not located within the ``main_shape``,
        these parts will be added to the resulting :class:`~.VMobject`.

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

    def __init__(
        self, main_shape: VMobject, *mobjects: VMobject, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.append_points(main_shape.points)
        sub_direction: Literal["CCW", "CW"] = (
            "CCW" if main_shape.get_direction() == "CW" else "CW"
        )
        for mobject in mobjects:
            self.append_points(mobject.force_direction(sub_direction).points)


class ConvexHull(Polygram):
    """Constructs a convex hull for a set of points in no particular order.

    Parameters
    ----------
    points
        The points to consider.
    tolerance
        The tolerance used by quickhull.
    kwargs
        Forwarded to the parent constructor.

    Examples
    --------
    .. manim:: ConvexHullExample
        :save_last_frame:
        :quality: high

        class ConvexHullExample(Scene):
            def construct(self):
                points = [
                    [-2.35, -2.25, 0],
                    [1.65, -2.25, 0],
                    [2.65, -0.25, 0],
                    [1.65, 1.75, 0],
                    [-0.35, 2.75, 0],
                    [-2.35, 0.75, 0],
                    [-0.35, -1.25, 0],
                    [0.65, -0.25, 0],
                    [-1.35, 0.25, 0],
                    [0.15, 0.75, 0]
                ]
                hull = ConvexHull(*points, color=BLUE)
                dots = VGroup(*[Dot(point) for point in points])
                self.add(hull)
                self.add(dots)
    """

    def __init__(
        self, *points: Point3DLike, tolerance: float = 1e-5, **kwargs: Any
    ) -> None:
        # Build Convex Hull
        array = np.array(points)[:, :2]
        hull = QuickHull(tolerance)
        hull.build(array)

        # Extract Vertices
        facets = set(hull.facets) - hull.removed
        facet = facets.pop()
        subfacets = list(facet.subfacets)
        while len(subfacets) <= len(facets):
            sf = subfacets[-1]
            (facet,) = hull.neighbors[sf] - {facet}
            (sf,) = facet.subfacets - {sf}
            subfacets.append(sf)

        # Setup Vertices as Point3D
        coordinates = np.vstack([sf.coordinates for sf in subfacets])
        vertices = np.hstack((coordinates, np.zeros((len(coordinates), 1))))

        # Call Polygram
        super().__init__(vertices, **kwargs)
