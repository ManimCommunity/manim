"""Three-dimensional mobjects."""

from __future__ import annotations

__all__ = [
    "ThreeDVMobject",
    "Surface",
    "Sphere",
    "Dot3D",
    "Cube",
    "Prism",
    "Cone",
    "Arrow3D",
    "Cylinder",
    "Line3D",
    "Torus",
]

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np

from manim._config import logger
from manim.appearance.material import Material, PbrMaterial
from manim.constants import *
from manim.mobject.mobject import *
from manim.mobject.three_d.mesh import ParametricSurface, PolygonFace
from manim.mobject.types.vectorized_mobject import VectorizedPoint, VGroup, VMobject
from manim.utils.color import (
    BLUE,
    BLUE_D,
    BLUE_E,
    LIGHT_GREY,
    WHITE,
    ManimColor,
    ParsableManimColor,
)
from manim.utils.space_ops import normalize, perpendicular_bisector, z_to_vector

if TYPE_CHECKING:
    from manim.mobject.graphing.coordinate_systems import ThreeDAxes
    from manim.typing import Point3D, Point3DLike, Vector3D, Vector3DLike


#: Default steps of a :class:`Sphere` around its axis and from pole to pole.
_SPHERE_RESOLUTION = (24, 12)
#: Default steps of a :class:`Torus` around its axis and around its tube.
_TORUS_RESOLUTION = (24, 24)


class ThreeDVMobject(VMobject):
    u_index: int
    v_index: int
    u1: float
    u2: float
    v1: float
    v2: float

    def __init__(self, shade_in_3d: bool = True, **kwargs: Any):
        super().__init__(shade_in_3d=shade_in_3d, **kwargs)


class Surface(ParametricSurface):
    """A parametric surface, by default of faces in a checkerboard pattern.

    The surface is a :class:`.ParametricSurface`: lit by a :class:`.ThreeDCamera`'s
    light through its ``material`` (``PbrMaterial()`` by default). With
    ``checkerboard_colors``, every face has its own corners, so faces keep their own
    color and are lit flat; without, neighboring faces share corners and the
    surface is lit smoothly.

    Parameters
    ----------
    func
        The function defining the :class:`Surface`.
    u_range
        The range of the ``u`` variable: ``(u_min, u_max)``.
    v_range
        The range of the ``v`` variable: ``(v_min, v_max)``.
    resolution
        The number of samples taken of the :class:`Surface`. A tuple can be
        used to define different resolutions for ``u`` and ``v`` respectively.
    fill_color
        The color of the :class:`Surface`. Ignored if ``checkerboard_colors``
        is set.
    fill_opacity
        The opacity of the :class:`Surface`, from 0 being fully transparent
        to 1 being fully opaque. Defaults to 1.
    checkerboard_colors
        The colors of alternating faces, or ``False`` for ``fill_color`` throughout.
    stroke_color
        Color of the outline of each face of :class:`Surface`.
    stroke_width
        Width of the outline of each face of :class:`Surface`. Defaults to 0.5.
    stroke_opacity
        Opacity of the outline of each face of :class:`Surface`.
    kwargs
        Further arguments of :class:`.ParametricSurface`, for example ``material``.

    Examples
    --------
    .. manim:: ParaSurface
        :save_last_frame:

        class ParaSurface(ThreeDScene):
            def func(self, u, v):
                return np.array([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), u])

            def construct(self):
                axes = ThreeDAxes(x_range=[-4,4], x_length=8)
                surface = Surface(
                    lambda u, v: axes.c2p(*self.func(u, v)),
                    u_range=[-PI, PI],
                    v_range=[0, TAU],
                    resolution=8,
                )
                self.set_camera_orientation(theta=70 * DEGREES, phi=75 * DEGREES)
                self.add(axes, surface)
    """

    def __init__(
        self,
        func: Callable[[float, float], np.ndarray],
        u_range: tuple[float, float] = (0, 1),
        v_range: tuple[float, float] = (0, 1),
        resolution: int | Sequence[int] = 32,
        fill_color: ParsableManimColor = BLUE_D,
        fill_opacity: float = 1.0,
        checkerboard_colors: Iterable[ParsableManimColor] | Literal[False] = [
            BLUE_D,
            BLUE_E,
        ],
        stroke_color: ParsableManimColor = LIGHT_GREY,
        stroke_width: float = 0.5,
        stroke_opacity: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.checkerboard_colors: list[ManimColor] | Literal[False]
        if checkerboard_colors is False:
            self.checkerboard_colors = checkerboard_colors
        else:
            self.checkerboard_colors = [ManimColor(i) for i in checkerboard_colors]
        # As for other mobjects, ``color`` is the fill color.
        fill_color = kwargs.pop("color", None) or fill_color
        super().__init__(
            func,
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,
            flat=bool(self.checkerboard_colors),
            color=fill_color,
            opacity=fill_opacity,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
            **kwargs,
        )
        if self.checkerboard_colors:
            self.set_fill_by_checkerboard(*self.checkerboard_colors)

    def set_color(
        self,
        color: ParsableManimColor = BLUE_D,
        alpha: float | None = None,
        family: bool = True,
    ) -> Self:
        """Color the faces and their outlines."""
        super().set_color(color, alpha, family)
        return self.set_stroke(color)

    def make_cap(self, radius: float) -> ParametricSurface:
        """A disc of ``radius`` around the origin in the xy-plane, as a cap of this
        surface (the base of a :class:`Cone`, the ends of a :class:`Cylinder`): its
        color, opacity and material, as many sectors as the surface has steps in
        ``v``, and no outlines. Subclasses can override it for other caps.
        """
        return ParametricSurface(
            lambda r, angle: np.array([r * np.cos(angle), r * np.sin(angle), 0.0]),
            u_range=(0, radius),
            v_range=(0, TAU),
            resolution=(1, self.resolution[1]),
            color=self.fill_color,
            opacity=self.fill_opacity,
            material=self.material,
            stroke_width=0,
        )

    def set_fill_by_value(  # type: ignore[override]
        self,
        axes: ThreeDAxes,
        colorscale: Iterable[ParsableManimColor]
        | Iterable[tuple[ParsableManimColor, float]]
        | None = None,
        axis: int = 2,
        **kwargs: Any,
    ) -> Self:
        """Color each face by its value along an axis: the value of the point
        halfway around the face's outline.

        Parameters
        ----------
        axes
            The axes for the parametric surface, which will be used to map
            axis-values to colors.
        colorscale
            A list of colors, ordered from lower axis-values to higher axis-values.
            If a list of tuples is passed containing colors paired with numbers,
            then those numbers will be used as the pivots.
        axis
            The chosen axis to use for the color mapping. (0 = x, 1 = y, 2 = z)

        Returns
        -------
        :class:`~.Surface`
            The parametric surface with a gradient applied by value. For chaining.

        Examples
        --------
        .. manim:: FillByValueExample
            :save_last_frame:

            class FillByValueExample(ThreeDScene):
                def construct(self):
                    resolution_fa = 8
                    self.set_camera_orientation(phi=75 * DEGREES, theta=-160 * DEGREES)
                    axes = ThreeDAxes(x_range=(0, 5, 1), y_range=(0, 5, 1), z_range=(-1, 1, 0.5))
                    def param_surface(u, v):
                        x = u
                        y = v
                        z = np.sin(x) * np.cos(y)
                        return z
                    surface_plane = Surface(
                        lambda u, v: axes.c2p(u, v, param_surface(u, v)),
                        resolution=(resolution_fa, resolution_fa),
                        v_range=[0, 5],
                        u_range=[0, 5],
                        )
                    surface_plane.set_style(fill_opacity=1)
                    surface_plane.set_fill_by_value(axes=axes, colorscale=[(RED, -0.5), (YELLOW, 0), (GREEN, 0.5)], axis=2)
                    self.add(axes, surface_plane)
        """
        if "colors" in kwargs and colorscale is None:
            colorscale = kwargs.pop("colors")
        if kwargs:
            raise ValueError(
                "Unsupported keyword argument(s): "
                f"{', '.join(str(key) for key in kwargs)}"
            )
        if colorscale is None:
            logger.warning(
                "The value passed to the colorscale keyword argument was None, "
                "the surface fill color has not been changed"
            )
            return self
        colorscale_list = list(colorscale)
        new_colors: list[ManimColor]
        if type(colorscale_list[0]) is tuple and len(colorscale_list[0]) == 2:
            pairs = cast("list[tuple[ParsableManimColor, float]]", colorscale_list)
            new_colors = [ManimColor(color) for color, _ in pairs]
            pivots = np.array([pivot for _, pivot in pairs], dtype=float)
        else:
            new_colors = [ManimColor(i) for i in colorscale_list]
            axis_range = [axes.x_range, axes.y_range, axes.z_range][axis]
            assert axis_range is not None
            pivot_min, pivot_max = axis_range[:2]
            pivots = np.linspace(pivot_min, pivot_max, len(new_colors))
        values = np.asarray(axes.point_to_coords(_halfway_around(self)))[:, axis]
        rgbs = np.array([color.to_rgb() for color in new_colors])
        colors = np.stack(
            [np.interp(values, pivots, rgbs[:, channel]) for channel in range(3)],
            axis=1,
        )
        opacity = self.fill_opacity
        fills = np.concatenate((colors, np.full((len(colors), 1), opacity)), axis=1)
        if not self.flat:
            # One color per face needs corners of its own, through the surface's
            # current points.
            grid = self._grid()
            self.flat = True
            self._build(grid)
        self.set_vertex_colors(np.repeat(fills, 4, axis=0))
        strokes = fills.copy()
        strokes[:, 3] = self.stroke_opacity
        # As each face's outline took its color, too.
        self.stroke_rgbas = strokes
        return self


def _halfway_around(surface: ParametricSurface) -> np.ndarray:
    """For each face, the point halfway around its outline."""
    corners = surface.points[surface.get_faces()]
    sides = np.linalg.norm(np.roll(corners, -1, axis=1) - corners, axis=2)
    ends = np.cumsum(sides, axis=1)
    halfway = ends[:, -1:] / 2
    side = np.minimum((ends < halfway).sum(axis=1), corners.shape[1] - 1)
    rows = np.arange(len(corners))
    start = corners[rows, side]
    end = corners[rows, (side + 1) % corners.shape[1]]
    before = np.where(side > 0, ends[rows, side - 1], 0)
    t = (halfway[:, 0] - before) / np.where(sides[rows, side] > 0, sides[rows, side], 1)
    return start + (end - start) * t[:, None]


# Specific shapes


class Sphere(Surface):
    """A three-dimensional sphere.

    Parameters
    ----------
    center
        Center of the :class:`Sphere`.
    radius
        The radius of the :class:`Sphere`.
    resolution
        The number of samples taken of the :class:`Sphere`. A tuple can be used
        to define different resolutions for ``u`` and ``v`` respectively.
    u_range
        The range of the ``u`` variable: ``(u_min, u_max)``.
    v_range
        The range of the ``v`` variable: ``(v_min, v_max)``.

    Examples
    --------

    .. manim:: ExampleSphere
        :save_last_frame:

        class ExampleSphere(ThreeDScene):
            def construct(self):
                self.set_camera_orientation(phi=PI / 6, theta=PI / 6)
                sphere1 = Sphere(center=(3, 0, 0), radius=1, resolution=(20, 20))
                sphere1.set_color(RED)
                self.add(sphere1)
                sphere2 = Sphere(center=(-1, -3, 0), radius=2, resolution=(18, 18))
                sphere2.set_color(GREEN)
                self.add(sphere2)
                sphere3 = Sphere(center=(-1, 2, 0), radius=2, resolution=(16, 16))
                sphere3.set_color(BLUE)
                self.add(sphere3)

    This example shows that overlapping spheres can intersect with rough transitions.

    .. manim:: ExampleSphereOverlap
        :save_last_frame:

        class ExampleSphereOverlap(ThreeDScene):
            def construct(self):
                self.set_camera_orientation(phi=PI / 4, theta=PI / 4)
                sphere1 = Sphere(center=(0, 0, 0), radius=1, resolution=(20, 20))
                sphere1.set_color(RED)
                self.add(sphere1)
                sphere2 = Sphere(center=(-0.5, -1, 0.5), radius=1.2, resolution=(20, 20))
                sphere2.set_color(GREEN)
                self.add(sphere2)
                sphere3 = Sphere(center=(1, -1, 0), radius=1.1, resolution=(20, 20))
                sphere3.set_color(BLUE)
                self.add(sphere3)

    In this example, by modifying ``u_range`` (the range of the azimuthal angle) and
    ``v_range`` (the range of the polar angle), it is possible to obtain a portion of a
    sphere:

    .. manim:: ExamplePartialSpheres
        :save_last_frame:

        class ExamplePartialSpheres(ThreeDScene):
            def construct(self):
                self.set_camera_orientation(phi=PI / 4)
                sphere1 = Sphere(
                    center=(-3, 0, 0),
                    resolution=(10, 20),
                    u_range=[TAU / 4, 3 * TAU / 4],
                )
                sphere1.set_color(RED)
                self.add(sphere1)
                sphere2 = Sphere(
                    center=(0, 0, 0),
                    resolution=(20, 10),
                    v_range=[0, TAU / 4],
                )
                sphere2.set_color(GREEN)
                self.add(sphere2)
                sphere3 = Sphere(
                    center=(3, 0, 0),
                    resolution=(5, 10),
                    u_range=[3 * TAU / 4, TAU],
                    v_range=[TAU / 4, TAU / 2],
                )
                sphere3.set_color(BLUE)
                self.add(sphere3)
    """

    def __init__(
        self,
        center: Point3DLike = ORIGIN,
        radius: float = 1,
        resolution: int | Sequence[int] | None = None,
        u_range: tuple[float, float] = (0, TAU),
        v_range: tuple[float, float] = (0, PI),
        **kwargs: Any,
    ) -> None:
        resolution = resolution if resolution is not None else _SPHERE_RESOLUTION

        self.radius = radius

        super().__init__(
            self.func,
            resolution=resolution,
            u_range=u_range,
            v_range=v_range,
            **kwargs,
        )

        self.shift(center)

    def func(self, u: float, v: float) -> Point3D:
        """The z values defining the :class:`Sphere` being plotted.

        Returns
        -------
        :class:`Point3D`
            The z values defining the :class:`Sphere`.
        """
        return self.radius * np.array(
            [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)],
        )


class Dot3D(Sphere):
    """A spherical dot.

    Parameters
    ----------
    point
        The location of the dot.
    radius
        The radius of the dot.
    color
        The color of the :class:`Dot3D`.
    resolution
        The number of samples taken of the :class:`Dot3D`. A tuple can be
        used to define different resolutions for ``u`` and ``v`` respectively.

    Examples
    --------

    .. manim:: Dot3DExample
        :save_last_frame:

        class Dot3DExample(ThreeDScene):
            def construct(self):
                self.set_camera_orientation(phi=75*DEGREES, theta=-45*DEGREES)

                axes = ThreeDAxes()
                dot_1 = Dot3D(point=axes.coords_to_point(0, 0, 1), color=RED)
                dot_2 = Dot3D(point=axes.coords_to_point(2, 0, 0), radius=0.1, color=BLUE)
                dot_3 = Dot3D(point=[0, 0, 0], radius=0.1, color=ORANGE)
                self.add(axes, dot_1, dot_2,dot_3)
    """

    def __init__(
        self,
        point: Point3D = ORIGIN,
        radius: float = DEFAULT_DOT_RADIUS,
        color: ParsableManimColor = WHITE,
        resolution: int | tuple[int, int] | None = (8, 8),
        **kwargs: Any,
    ) -> None:
        super().__init__(center=point, radius=radius, resolution=resolution, **kwargs)
        self.set_color(color)


class Cube(VGroup):
    """A three-dimensional cube.

    Parameters
    ----------
    side_length
        Length of each side of the :class:`Cube`.
    fill_opacity
        The opacity of the :class:`Cube`, from 0 being fully transparent to 1 being
        fully opaque. Defaults to 0.75.
    fill_color
        The color of the :class:`Cube`.
    stroke_width
        The width of the stroke surrounding each face of the :class:`Cube`.

    Examples
    --------

    .. manim:: CubeExample
        :save_last_frame:

        class CubeExample(ThreeDScene):
            def construct(self):
                self.set_camera_orientation(phi=75*DEGREES, theta=-45*DEGREES)

                axes = ThreeDAxes()
                cube = Cube(side_length=3, fill_opacity=0.7, fill_color=BLUE)
                self.add(cube)
    """

    def __init__(
        self,
        side_length: float = 2,
        fill_opacity: float = 0.75,
        fill_color: ParsableManimColor = BLUE,
        stroke_width: float = 0,
        material: Material | None = None,
        **kwargs: Any,
    ) -> None:
        self.side_length = side_length
        self._face_material = PbrMaterial() if material is None else material
        super().__init__(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs,
        )

    def generate_points(self) -> Self:
        """Creates the sides of the :class:`Cube`: square faces, lit by the
        ``material`` (:class:`.PbrMaterial` by default).
        """
        half = self.side_length / 2
        square = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        for vect in IN, OUT, LEFT, RIGHT, UP, DOWN:
            corners = np.column_stack([square, np.full(4, half)])
            corners = corners @ np.asarray(z_to_vector(vect)).T
            self.add(PolygonFace(*corners, material=self._face_material))
        return self


class Prism(Cube):
    """A right rectangular prism (or rectangular cuboid).
    Defined by the length of each side in ``[x, y, z]`` format.

    Parameters
    ----------
    dimensions
        Dimensions of the :class:`Prism` in ``[x, y, z]`` format.

    Examples
    --------

    .. manim:: ExamplePrism
        :save_last_frame:

        class ExamplePrism(ThreeDScene):
            def construct(self):
                self.set_camera_orientation(phi=60 * DEGREES, theta=150 * DEGREES)
                prismSmall = Prism(dimensions=[1, 2, 3]).rotate(PI / 2)
                prismLarge = Prism(dimensions=[1.5, 3, 4.5]).move_to([2, 0, 0])
                self.add(prismSmall, prismLarge)
    """

    def __init__(
        self,
        dimensions: Vector3DLike = [3, 2, 1],
        **kwargs: Any,
    ) -> None:
        self.dimensions = dimensions
        super().__init__(**kwargs)

    def generate_points(self) -> Self:
        """Creates the sides of the :class:`Prism`."""
        super().generate_points()
        for dim, value in enumerate(self.dimensions):
            self.rescale_to_fit(value, dim, stretch=True)
        return self


class Cone(Surface):
    """A circular cone.
    Can be defined using 2 parameters: its height, and its base radius.
    The polar angle, theta, can be calculated using arctan(base_radius /
    height) The spherical radius, r, is calculated using the pythagorean
    theorem.

    Parameters
    ----------
    base_radius
        The base radius from which the cone tapers.
    height
        The height measured from the plane formed by the base_radius to
        the apex of the cone.
    direction
        The direction of the apex.
    show_base
        Whether to show the base plane or not.
    v_range
        The azimuthal angle to start and end at.
    u_min
        The radius at the apex.
    checkerboard_colors
        Show checkerboard grid texture on the cone.

    Examples
    --------
    .. manim:: ExampleCone
        :save_last_frame:

        class ExampleCone(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                cone = Cone(direction=X_AXIS+Y_AXIS+2*Z_AXIS, resolution=8)
                self.set_camera_orientation(phi=5*PI/11, theta=PI/9)
                self.add(axes, cone)
    """

    def __init__(
        self,
        base_radius: float = 1,
        height: float = 1,
        direction: Vector3DLike = Z_AXIS,
        show_base: bool = False,
        v_range: tuple[float, float] = (0, TAU),
        u_min: float = 0,
        checkerboard_colors: Iterable[ParsableManimColor] | Literal[False] = False,
        **kwargs: Any,
    ) -> None:
        self.direction = np.array(direction)
        self.theta = PI - np.arctan(base_radius / height)

        super().__init__(
            self.func,
            v_range=v_range,
            u_range=(u_min, np.sqrt(base_radius**2 + height**2)),
            checkerboard_colors=checkerboard_colors,
            **kwargs,
        )
        # used for rotations
        self.new_height = height
        self._current_theta = 0
        self._current_phi = 0
        self.base_circle = self.make_cap(base_radius)
        self.base_circle.shift(height * IN)
        self._set_start_and_end_attributes(direction)
        if show_base:
            self.add(self.base_circle)

        self._rotate_to_direction()

    def func(self, u: float, v: float) -> Point3D:
        """Converts from spherical coordinates to cartesian.

        Parameters
        ----------
        u
            The radius.
        v
            The azimuthal angle.

        Returns
        -------
        :class:`numpy.array`
            Points defining the :class:`Cone`.
        """
        r = u
        phi = v
        return np.array(
            [
                r * np.sin(self.theta) * np.cos(phi),
                r * np.sin(self.theta) * np.sin(phi),
                r * np.cos(self.theta),
            ],
        )

    def get_start(self) -> Point3D:
        return self.start_point.get_center()

    def get_end(self) -> Point3D:
        return self.end_point.get_center()

    def _rotate_to_direction(self) -> None:
        x, y, z = self.direction

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) if r > 0 else 0

        if x == 0:
            if y == 0:  # along the z axis
                phi = 0
            else:
                phi = np.arctan(np.inf)
                if y < 0:
                    phi += PI
        else:
            phi = np.arctan(y / x)
        if x < 0:
            phi += PI

        # Undo old rotation (in reverse order)
        self.rotate(-self._current_phi, Z_AXIS, about_point=ORIGIN)
        self.rotate(-self._current_theta, Y_AXIS, about_point=ORIGIN)

        # Do new rotation
        self.rotate(theta, Y_AXIS, about_point=ORIGIN)
        self.rotate(phi, Z_AXIS, about_point=ORIGIN)

        # Store values
        self._current_theta = theta
        self._current_phi = phi

    def set_direction(self, direction: Vector3DLike) -> Self:
        """Changes the direction of the apex of the :class:`Cone`.

        Parameters
        ----------
        direction
            The direction of the apex.
        """
        self.direction = np.array(direction)
        self._rotate_to_direction()
        return self

    def get_direction(self) -> Vector3D:
        """Returns the current direction of the apex of the :class:`Cone`.

        Returns
        -------
        direction : :class:`numpy.array`
            The direction of the apex.
        """
        return self.direction

    def _set_start_and_end_attributes(self, direction: Vector3D) -> None:
        normalized_direction = direction * np.linalg.norm(direction)

        start = self.base_circle.get_center()
        end = start + normalized_direction * self.new_height
        self.start_point = VectorizedPoint(start)
        self.end_point = VectorizedPoint(end)
        self.add(self.start_point, self.end_point)


class Cylinder(Surface):
    """A cylinder, defined by its height, radius and direction,

    Parameters
    ----------
    radius
        The radius of the cylinder.
    height
        The height of the cylinder.
    direction
        The direction of the central axis of the cylinder.
    v_range
        The height along the height axis (given by direction) to start and end on.
    show_ends
        Whether to show the end caps or not.
    resolution
        The number of samples taken of the :class:`Cylinder`. A tuple can be used
        to define different resolutions for ``u`` and ``v`` respectively.

    Examples
    --------
    .. manim:: ExampleCylinder
        :save_last_frame:

        class ExampleCylinder(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                cylinder = Cylinder(radius=2, height=3)
                self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                self.add(axes, cylinder)
    """

    def __init__(
        self,
        radius: float = 1,
        height: float = 2,
        direction: Vector3DLike = Z_AXIS,
        v_range: tuple[float, float] = (0, TAU),
        show_ends: bool = True,
        resolution: int | tuple[int, int] = (24, 24),
        **kwargs: Any,
    ) -> None:
        self._height = height
        self.radius = radius
        super().__init__(
            self.func,
            resolution=resolution,
            u_range=(-self._height / 2, self._height / 2),
            v_range=v_range,
            **kwargs,
        )
        if show_ends:
            self.add_bases()
        self._current_phi = 0
        self._current_theta = 0
        self.set_direction(direction)

    def func(self, u: float, v: float) -> np.ndarray:
        """Converts from cylindrical coordinates to cartesian.

        Parameters
        ----------
        u
            The height.
        v
            The azimuthal angle.

        Returns
        -------
        :class:`numpy.ndarray`
            Points defining the :class:`Cylinder`.
        """
        height = u
        phi = v
        r = self.radius
        return np.array([r * np.cos(phi), r * np.sin(phi), height])

    def add_bases(self) -> Self:
        """Adds the end caps of the cylinder: discs lit as the side is."""
        self.base_top = self.make_cap(self.radius)
        self.base_top.shift(self.u_range[1] * IN)
        self.base_bottom = self.make_cap(self.radius)
        self.base_bottom.shift(self.u_range[0] * IN)
        self.add(self.base_top, self.base_bottom)
        return self

    def _rotate_to_direction(self) -> None:
        x, y, z = self.direction

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) if r > 0 else 0

        if x == 0:
            if y == 0:  # along the z axis
                phi = 0
            else:  # along the x axis
                phi = np.arctan(np.inf)
                if y < 0:
                    phi += PI
        else:
            phi = np.arctan(y / x)
        if x < 0:
            phi += PI

        # undo old rotation (in reverse direction)
        self.rotate(-self._current_phi, Z_AXIS, about_point=ORIGIN)
        self.rotate(-self._current_theta, Y_AXIS, about_point=ORIGIN)

        # do new rotation
        self.rotate(theta, Y_AXIS, about_point=ORIGIN)
        self.rotate(phi, Z_AXIS, about_point=ORIGIN)

        # store new values
        self._current_theta = theta
        self._current_phi = phi

    def set_direction(self, direction: Vector3DLike) -> Self:
        """Sets the direction of the central axis of the :class:`Cylinder`.

        Parameters
        ----------
        direction : :class:`numpy.array`
            The direction of the central axis of the :class:`Cylinder`.
        """
        # if get_norm(direction) is get_norm(self.direction):
        #     pass
        self.direction = direction
        self._rotate_to_direction()
        return self

    def get_direction(self) -> np.ndarray:
        """Returns the direction of the central axis of the :class:`Cylinder`.

        Returns
        -------
        direction : :class:`numpy.array`
            The direction of the central axis of the :class:`Cylinder`.
        """
        return self.direction


class Line3D(Cylinder):
    """A cylindrical line, for use in ThreeDScene.

    Parameters
    ----------
    start
        The start point of the line.
    end
        The end point of the line.
    thickness
        The thickness of the line.
    color
        The color of the line.
    resolution
        The resolution of the line.
        By default this value is the number of points the line will sampled at.
        If you want the line to also come out checkered, use a tuple.
        For example, for a line made of 24 points with 4 checker points on each
        cylinder, pass the tuple (4, 24).

    Examples
    --------
    .. manim:: ExampleLine3D
        :save_last_frame:

        class ExampleLine3D(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                line = Line3D(start=np.array([0, 0, 0]), end=np.array([2, 2, 2]))
                self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                self.add(axes, line)
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        thickness: float = 0.02,
        color: ParsableManimColor | None = None,
        resolution: int | tuple[int, int] = 24,
        **kwargs: Any,
    ):
        self.thickness = thickness
        self.resolution: tuple[int, int] = (
            (2, resolution) if isinstance(resolution, int) else resolution
        )

        start = np.array(start, dtype=np.float64)
        end = np.array(end, dtype=np.float64)

        self.set_start_and_end_attrs(start, end, **kwargs)
        if color is not None:
            self.set_color(color)

    def set_start_and_end_attrs(
        self, start: Point3DLike, end: Point3DLike, **kwargs: Any
    ) -> Self:
        """Sets the start and end points of the line.

        If either ``start`` or ``end`` are :class:`Mobjects <.Mobject>`,
        this gives their centers.

        Parameters
        ----------
        start
            Starting point or :class:`Mobject`.
        end
            Ending point or :class:`Mobject`.
        """
        rough_start = self.pointify(start)
        rough_end = self.pointify(end)
        self.vect = rough_end - rough_start
        self.length = np.linalg.norm(self.vect)
        self.direction: Vector3D = normalize(self.vect)
        # Now that we know the direction between them,
        # we can the appropriate boundary point from
        # start and end, if they're mobjects
        self.start = self.pointify(start, self.direction)
        self.end = self.pointify(end, -self.direction)
        super().__init__(
            height=np.linalg.norm(self.vect),
            radius=self.thickness,
            direction=self.direction,
            resolution=self.resolution,
            **kwargs,
        )
        self.shift((self.start + self.end) / 2)
        return self

    def pointify(
        self,
        mob_or_point: Mobject | Point3DLike,
        direction: Vector3DLike | None = None,
    ) -> Point3D:
        """Gets a point representing the center of the :class:`Mobjects <.Mobject>`.

        Parameters
        ----------
        mob_or_point
            :class:`Mobjects <.Mobject>` or point whose center should be returned.
        direction
            If an edge of a :class:`Mobjects <.Mobject>` should be returned, the direction of the edge.

        Returns
        -------
        :class:`numpy.array`
            Center of the :class:`Mobjects <.Mobject>` or point, or edge if direction is given.
        """
        if isinstance(mob_or_point, Mobject):
            mob = mob_or_point
            if direction is None:
                return mob.get_center()
            else:
                return mob.get_boundary_point(direction)
        return np.array(mob_or_point)

    def get_start(self) -> Point3D:
        """Returns the starting point of the :class:`Line3D`.

        Returns
        -------
        start : :class:`numpy.array`
            Starting point of the :class:`Line3D`.
        """
        return self.start

    def get_end(self) -> Point3D:
        """Returns the ending point of the :class:`Line3D`.

        Returns
        -------
        end : :class:`numpy.array`
            Ending point of the :class:`Line3D`.
        """
        return self.end

    @classmethod
    def parallel_to(
        cls,
        line: Line3D,
        point: Point3DLike = ORIGIN,
        length: float = 5,
        **kwargs: Any,
    ) -> Line3D:
        """Returns a line parallel to another line going through
        a given point.

        Parameters
        ----------
        line
            The line to be parallel to.
        point
            The point to pass through.
        length
            Length of the parallel line.
        kwargs
            Additional parameters to be passed to the class.

        Returns
        -------
        :class:`Line3D`
            Line parallel to ``line``.

        Examples
        --------
        .. manim:: ParallelLineExample
            :save_last_frame:

            class ParallelLineExample(ThreeDScene):
                def construct(self):
                    self.set_camera_orientation(PI / 3, -PI / 4)
                    ax = ThreeDAxes((-5, 5), (-5, 5), (-5, 5), 10, 10, 10)
                    line1 = Line3D(RIGHT * 2, UP + OUT, color=RED)
                    line2 = Line3D.parallel_to(line1, color=YELLOW)
                    self.add(ax, line1, line2)
        """
        np_point = np.asarray(point)
        vect = normalize(line.vect)
        return cls(
            np_point + vect * length / 2,
            np_point - vect * length / 2,
            **kwargs,
        )

    @classmethod
    def perpendicular_to(
        cls,
        line: Line3D,
        point: Point3DLike = ORIGIN,
        length: float = 5,
        **kwargs: Any,
    ) -> Line3D:
        """Returns a line perpendicular to another line going through
        a given point.

        Parameters
        ----------
        line
            The line to be perpendicular to.
        point
            The point to pass through.
        length
            Length of the perpendicular line.
        kwargs
            Additional parameters to be passed to the class.

        Returns
        -------
        :class:`Line3D`
            Line perpendicular to ``line``.

        Examples
        --------
        .. manim:: PerpLineExample
            :save_last_frame:

            class PerpLineExample(ThreeDScene):
                def construct(self):
                    self.set_camera_orientation(PI / 3, -PI / 4)
                    ax = ThreeDAxes((-5, 5), (-5, 5), (-5, 5), 10, 10, 10)
                    line1 = Line3D(RIGHT * 2, UP + OUT, color=RED)
                    line2 = Line3D.perpendicular_to(line1, color=BLUE)
                    self.add(ax, line1, line2)
        """
        np_point = np.asarray(point)

        norm = np.cross(line.vect, np_point - line.start)
        if all(np.linalg.norm(norm) == np.zeros(3)):
            raise ValueError("Could not find the perpendicular.")

        start, end = perpendicular_bisector([line.start, line.end], norm)
        vect = normalize(end - start)
        return cls(
            np_point + vect * length / 2,
            np_point - vect * length / 2,
            **kwargs,
        )


class Arrow3D(Line3D):
    """An arrow made out of a cylindrical line and a conical tip.

    Parameters
    ----------
    start
        The start position of the arrow.
    end
        The end position of the arrow.
    thickness
        The thickness of the arrow.
    height
        The height of the conical tip.
    base_radius
        The base radius of the conical tip.
    color
        The color of the arrow.
    resolution
        The resolution of the arrow line.

    Examples
    --------
    .. manim:: ExampleArrow3D
        :save_last_frame:

        class ExampleArrow3D(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                arrow = Arrow3D(
                    start=np.array([0, 0, 0]),
                    end=np.array([2, 2, 2]),
                    resolution=8
                )
                self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                self.add(axes, arrow)
    """

    def __init__(
        self,
        start: Point3DLike = LEFT,
        end: Point3DLike = RIGHT,
        thickness: float = 0.02,
        height: float = 0.3,
        base_radius: float = 0.08,
        color: ParsableManimColor = WHITE,
        resolution: int | tuple[int, int] = 24,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            start=start,
            end=end,
            thickness=thickness,
            color=color,
            resolution=resolution,
            **kwargs,
        )

        self.length = np.linalg.norm(self.vect)
        self.set_start_and_end_attrs(
            self.start,
            self.end - height * self.direction,
            **kwargs,
        )
        self.cone = Cone(
            direction=self.direction,
            base_radius=base_radius,
            height=height,
            **kwargs,
        )
        np_end = np.asarray(end, dtype=np.float64)
        self.cone.shift(np_end)
        self.end_point = VectorizedPoint(np_end)
        self.add(self.end_point, self.cone)
        self.set_color(color)

    def get_end(self) -> np.ndarray:
        return self.end_point.get_center()


class Torus(Surface):
    """A torus.

    Parameters
    ----------
    major_radius
        Distance from the center of the tube to the center of the torus.
    minor_radius
        Radius of the tube.
    u_range
        The range of the ``u`` variable: ``(u_min, u_max)``.
    v_range
        The range of the ``v`` variable: ``(v_min, v_max)``.
    resolution
        The number of samples taken of the :class:`Torus`. A tuple can be
        used to define different resolutions for ``u`` and ``v`` respectively.

    Examples
    --------
    .. manim :: ExampleTorus
        :save_last_frame:

        class ExampleTorus(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                torus = Torus()
                self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                self.add(axes, torus)
    """

    def __init__(
        self,
        major_radius: float = 3,
        minor_radius: float = 1,
        u_range: tuple[float, float] = (0, TAU),
        v_range: tuple[float, float] = (0, TAU),
        resolution: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        resolution = resolution if resolution is not None else _TORUS_RESOLUTION

        self.R = major_radius
        self.r = minor_radius
        super().__init__(
            self.func,
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,
            **kwargs,
        )

    def func(self, u: float, v: float) -> Point3D:
        """The z values defining the :class:`Torus` being plotted.

        Returns
        -------
        :class:`numpy.ndarray`
            The z values defining the :class:`Torus`.
        """
        P = np.array([np.cos(u), np.sin(u), 0])
        return (self.R - self.r * np.cos(v)) * P - self.r * np.sin(v) * OUT
