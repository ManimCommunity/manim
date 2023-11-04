"""Three-dimensional mobjects."""

from __future__ import annotations

from manim.typing import Point3D, Vector3
from manim.utils.color import BLUE, BLUE_D, BLUE_E, LIGHT_GREY, WHITE, interpolate_color

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

from typing import Any, Callable, Iterable, Sequence

import numpy as np
from typing_extensions import Self

from manim import config, logger
from manim.constants import *
from manim.mobject.geometry.arc import Circle
from manim.mobject.geometry.polygram import Square
from manim.mobject.mobject import *
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.color import (
    BLUE,
    BLUE_D,
    BLUE_E,
    LIGHT_GREY,
    WHITE,
    ManimColor,
    ParsableManimColor,
    interpolate_color,
)
from manim.utils.iterables import tuplify
from manim.utils.space_ops import normalize, perpendicular_bisector, z_to_vector


class ThreeDVMobject(VMobject, metaclass=ConvertToOpenGL):
    def __init__(self, shade_in_3d: bool = True, **kwargs):
        super().__init__(shade_in_3d=shade_in_3d, **kwargs)


class Surface(VGroup, metaclass=ConvertToOpenGL):
    """Creates a Parametric Surface using a checkerboard pattern.

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
        ng individual faces alternating colors. Overrides ``fill_color``.
    stroke_color
        Color of the stroke surrounding each face of :class:`Surface`.
    stroke_width
        Width of the stroke surrounding each face of :class:`Surface`.
        Defaults to 0.5.
    should_make_jagged
        Changes the anchor mode of the Bézier curves from smooth to jagged.
        Defaults to ``False``.

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
        u_range: Sequence[float] = [0, 1],
        v_range: Sequence[float] = [0, 1],
        resolution: Sequence[int] = 32,
        surface_piece_config: dict = {},
        fill_color: ParsableManimColor = BLUE_D,
        fill_opacity: float = 1.0,
        checkerboard_colors: Sequence[ParsableManimColor] | bool = [BLUE_D, BLUE_E],
        stroke_color: ParsableManimColor = LIGHT_GREY,
        stroke_width: float = 0.5,
        should_make_jagged: bool = False,
        pre_function_handle_to_anchor_scale_factor: float = 0.00001,
        **kwargs: Any,
    ) -> None:
        self.u_range = u_range
        self.v_range = v_range
        super().__init__(**kwargs)
        self.resolution = resolution
        self.surface_piece_config = surface_piece_config
        self.fill_color: ManimColor = ManimColor(fill_color)
        self.fill_opacity = fill_opacity
        if checkerboard_colors:
            self.checkerboard_colors: list[ManimColor] = [
                ManimColor(x) for x in checkerboard_colors
            ]
        else:
            self.checkerboard_colors = checkerboard_colors
        self.stroke_color: ManimColor = ManimColor(stroke_color)
        self.stroke_width = stroke_width
        self.should_make_jagged = should_make_jagged
        self.pre_function_handle_to_anchor_scale_factor = (
            pre_function_handle_to_anchor_scale_factor
        )
        self._func = func
        self._setup_in_uv_space()
        self.apply_function(lambda p: func(p[0], p[1]))
        if self.should_make_jagged:
            self.make_jagged()

    def func(self, u: float, v: float) -> np.ndarray:
        return self._func(u, v)

    def _get_u_values_and_v_values(self) -> tuple[np.ndarray, np.ndarray]:
        res = tuplify(self.resolution)
        if len(res) == 1:
            u_res = v_res = res[0]
        else:
            u_res, v_res = res

        u_values = np.linspace(*self.u_range, u_res + 1)
        v_values = np.linspace(*self.v_range, v_res + 1)

        return u_values, v_values

    def _setup_in_uv_space(self) -> None:
        u_values, v_values = self._get_u_values_and_v_values()
        faces = VGroup()
        for i in range(len(u_values) - 1):
            for j in range(len(v_values) - 1):
                u1, u2 = u_values[i : i + 2]
                v1, v2 = v_values[j : j + 2]
                face = ThreeDVMobject()
                face.set_points_as_corners(
                    [
                        [u1, v1, 0],
                        [u2, v1, 0],
                        [u2, v2, 0],
                        [u1, v2, 0],
                        [u1, v1, 0],
                    ],
                )
                faces.add(face)
                face.u_index = i
                face.v_index = j
                face.u1 = u1
                face.u2 = u2
                face.v1 = v1
                face.v2 = v2
        faces.set_fill(color=self.fill_color, opacity=self.fill_opacity)
        faces.set_stroke(
            color=self.stroke_color,
            width=self.stroke_width,
            opacity=self.stroke_opacity,
        )
        self.add(*faces)
        if self.checkerboard_colors:
            self.set_fill_by_checkerboard(*self.checkerboard_colors)

    def set_fill_by_checkerboard(
        self, *colors: Iterable[ParsableManimColor], opacity: float | None = None
    ) -> Self:
        """Sets the fill_color of each face of :class:`Surface` in
        an alternating pattern.

        Parameters
        ----------
        colors
            List of colors for alternating pattern.
        opacity
            The fill_opacity of :class:`Surface`, from 0 being fully transparent
            to 1 being fully opaque.

        Returns
        -------
        :class:`~.Surface`
            The parametric surface with an alternating pattern.
        """
        n_colors = len(colors)
        for face in self:
            c_index = (face.u_index + face.v_index) % n_colors
            face.set_fill(colors[c_index], opacity=opacity)
        return self

    def set_fill_by_value(
        self,
        axes: Mobject,
        colorscale: list[ParsableManimColor] | ParsableManimColor | None = None,
        axis: int = 2,
        **kwargs,
    ) -> Self:
        """Sets the color of each mobject of a parametric surface to a color
        relative to its axis-value.

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

        ranges = [axes.x_range, axes.y_range, axes.z_range]

        if type(colorscale[0]) is tuple:
            new_colors, pivots = [
                [i for i, j in colorscale],
                [j for i, j in colorscale],
            ]
        else:
            new_colors = colorscale

            pivot_min = ranges[axis][0]
            pivot_max = ranges[axis][1]
            pivot_frequency = (pivot_max - pivot_min) / (len(new_colors) - 1)
            pivots = np.arange(
                start=pivot_min,
                stop=pivot_max + pivot_frequency,
                step=pivot_frequency,
            )

        for mob in self.family_members_with_points():
            axis_value = axes.point_to_coords(mob.get_midpoint())[axis]
            if axis_value <= pivots[0]:
                mob.set_color(new_colors[0])
            elif axis_value >= pivots[-1]:
                mob.set_color(new_colors[-1])
            else:
                for i, pivot in enumerate(pivots):
                    if pivot > axis_value:
                        color_index = (axis_value - pivots[i - 1]) / (
                            pivots[i] - pivots[i - 1]
                        )
                        color_index = min(color_index, 1)
                        mob_color = interpolate_color(
                            new_colors[i - 1],
                            new_colors[i],
                            color_index,
                        )
                        if config.renderer == RendererType.OPENGL:
                            mob.set_color(mob_color, recurse=False)
                        elif config.renderer == RendererType.CAIRO:
                            mob.set_color(mob_color, family=False)
                        break

        return self


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
                sphere1 = Sphere(
                    center=(3, 0, 0),
                    radius=1,
                    resolution=(20, 20),
                    u_range=[0.001, PI - 0.001],
                    v_range=[0, TAU]
                )
                sphere1.set_color(RED)
                self.add(sphere1)
                sphere2 = Sphere(center=(-1, -3, 0), radius=2, resolution=(18, 18))
                sphere2.set_color(GREEN)
                self.add(sphere2)
                sphere3 = Sphere(center=(-1, 2, 0), radius=2, resolution=(16, 16))
                sphere3.set_color(BLUE)
                self.add(sphere3)
    """

    def __init__(
        self,
        center: Point3D = ORIGIN,
        radius: float = 1,
        resolution: Sequence[int] | None = None,
        u_range: Sequence[float] = (0, TAU),
        v_range: Sequence[float] = (0, PI),
        **kwargs,
    ) -> None:
        if config.renderer == RendererType.OPENGL:
            res_value = (101, 51)
        elif config.renderer == RendererType.CAIRO:
            res_value = (24, 12)
        else:
            raise Exception("Unknown renderer")

        resolution = resolution if resolution is not None else res_value

        self.radius = radius

        super().__init__(
            self.func,
            resolution=resolution,
            u_range=u_range,
            v_range=v_range,
            **kwargs,
        )

        self.shift(center)

    def func(self, u: float, v: float) -> np.ndarray:
        """The z values defining the :class:`Sphere` being plotted.

        Returns
        -------
        :class:`numpy.array`
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
        point: list | np.ndarray = ORIGIN,
        radius: float = DEFAULT_DOT_RADIUS,
        color: ParsableManimColor = WHITE,
        resolution: tuple[int, int] = (8, 8),
        **kwargs,
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
        **kwargs,
    ) -> None:
        self.side_length = side_length
        super().__init__(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs,
        )

    def generate_points(self) -> None:
        """Creates the sides of the :class:`Cube`."""
        for vect in IN, OUT, LEFT, RIGHT, UP, DOWN:
            face = Square(
                side_length=self.side_length,
                shade_in_3d=True,
            )
            face.flip()
            face.shift(self.side_length * OUT / 2.0)
            face.apply_matrix(z_to_vector(vect))

            self.add(face)

    init_points = generate_points


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
        self, dimensions: tuple[float, float, float] | np.ndarray = [3, 2, 1], **kwargs
    ) -> None:
        self.dimensions = dimensions
        super().__init__(**kwargs)

    def generate_points(self) -> None:
        """Creates the sides of the :class:`Prism`."""
        super().generate_points()
        for dim, value in enumerate(self.dimensions):
            self.rescale_to_fit(value, dim, stretch=True)


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
        direction: np.ndarray = Z_AXIS,
        show_base: bool = False,
        v_range: Sequence[float] = [0, TAU],
        u_min: float = 0,
        checkerboard_colors: bool = False,
        **kwargs: Any,
    ) -> None:
        self.direction = direction
        self.theta = PI - np.arctan(base_radius / height)

        super().__init__(
            self.func,
            v_range=v_range,
            u_range=[u_min, np.sqrt(base_radius**2 + height**2)],
            checkerboard_colors=checkerboard_colors,
            **kwargs,
        )
        # used for rotations
        self._current_theta = 0
        self._current_phi = 0

        if show_base:
            self.base_circle = Circle(
                radius=base_radius,
                color=self.fill_color,
                fill_opacity=self.fill_opacity,
                stroke_width=0,
            )
            self.base_circle.shift(height * IN)
            self.add(self.base_circle)

        self._rotate_to_direction()

    def func(self, u: float, v: float) -> np.ndarray:
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

    def _rotate_to_direction(self) -> None:
        x, y, z = self.direction

        r = np.sqrt(x**2 + y**2 + z**2)
        if r > 0:
            theta = np.arccos(z / r)
        else:
            theta = 0

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

    def set_direction(self, direction: np.ndarray) -> None:
        """Changes the direction of the apex of the :class:`Cone`.

        Parameters
        ----------
        direction
            The direction of the apex.
        """
        self.direction = direction
        self._rotate_to_direction()

    def get_direction(self) -> np.ndarray:
        """Returns the current direction of the apex of the :class:`Cone`.

        Returns
        -------
        direction : :class:`numpy.array`
            The direction of the apex.
        """
        return self.direction


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
        direction: np.ndarray = Z_AXIS,
        v_range: Sequence[float] = [0, TAU],
        show_ends: bool = True,
        resolution: Sequence[int] = (24, 24),
        **kwargs,
    ) -> None:
        self._height = height
        self.radius = radius
        super().__init__(
            self.func,
            resolution=resolution,
            u_range=[-self._height / 2, self._height / 2],
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

    def add_bases(self) -> None:
        """Adds the end caps of the cylinder."""
        if config.renderer == RendererType.OPENGL:
            color = self.color
            opacity = self.opacity
        elif config.renderer == RendererType.CAIRO:
            color = self.fill_color
            opacity = self.fill_opacity

        self.base_top = Circle(
            radius=self.radius,
            color=color,
            fill_opacity=opacity,
            shade_in_3d=True,
            stroke_width=0,
        )
        self.base_top.shift(self.u_range[1] * IN)
        self.base_bottom = Circle(
            radius=self.radius,
            color=color,
            fill_opacity=opacity,
            shade_in_3d=True,
            stroke_width=0,
        )
        self.base_bottom.shift(self.u_range[0] * IN)
        self.add(self.base_top, self.base_bottom)

    def _rotate_to_direction(self) -> None:
        x, y, z = self.direction

        r = np.sqrt(x**2 + y**2 + z**2)
        if r > 0:
            theta = np.arccos(z / r)
        else:
            theta = 0

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

    def set_direction(self, direction: np.ndarray) -> None:
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
        start: np.ndarray = LEFT,
        end: np.ndarray = RIGHT,
        thickness: float = 0.02,
        color: ParsableManimColor | None = None,
        **kwargs,
    ):
        self.thickness = thickness
        self.set_start_and_end_attrs(start, end, **kwargs)
        if color is not None:
            self.set_color(color)

    def set_start_and_end_attrs(
        self, start: np.ndarray, end: np.ndarray, **kwargs
    ) -> None:
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
        self.direction = normalize(self.vect)
        # Now that we know the direction between them,
        # we can the appropriate boundary point from
        # start and end, if they're mobjects
        self.start = self.pointify(start, self.direction)
        self.end = self.pointify(end, -self.direction)
        super().__init__(
            height=np.linalg.norm(self.vect),
            radius=self.thickness,
            direction=self.direction,
            **kwargs,
        )
        self.shift((self.start + self.end) / 2)

    def pointify(
        self,
        mob_or_point: Mobject | Point3D,
        direction: Vector3 = None,
    ) -> np.ndarray:
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
        if isinstance(mob_or_point, (Mobject, OpenGLMobject)):
            mob = mob_or_point
            if direction is None:
                return mob.get_center()
            else:
                return mob.get_boundary_point(direction)
        return np.array(mob_or_point)

    def get_start(self) -> np.ndarray:
        """Returns the starting point of the :class:`Line3D`.

        Returns
        -------
        start : :class:`numpy.array`
            Starting point of the :class:`Line3D`.
        """
        return self.start

    def get_end(self) -> np.ndarray:
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
        point: Vector3 = ORIGIN,
        length: float = 5,
        **kwargs,
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
        point = np.array(point)
        vect = normalize(line.vect)
        return cls(
            point + vect * length / 2,
            point - vect * length / 2,
            **kwargs,
        )

    @classmethod
    def perpendicular_to(
        cls,
        line: Line3D,
        point: Vector3 = ORIGIN,
        length: float = 5,
        **kwargs,
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
        point = np.array(point)

        norm = np.cross(line.vect, point - line.start)
        if all(np.linalg.norm(norm) == np.zeros(3)):
            raise ValueError("Could not find the perpendicular.")

        start, end = perpendicular_bisector([line.start, line.end], norm)
        vect = normalize(end - start)
        return cls(
            point + vect * length / 2,
            point - vect * length / 2,
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
        start: np.ndarray = LEFT,
        end: np.ndarray = RIGHT,
        thickness: float = 0.02,
        height: float = 0.3,
        base_radius: float = 0.08,
        color: ParsableManimColor = WHITE,
        **kwargs,
    ) -> None:
        super().__init__(
            start=start, end=end, thickness=thickness, color=color, **kwargs
        )

        self.length = np.linalg.norm(self.vect)
        self.set_start_and_end_attrs(
            self.start,
            self.end - height * self.direction,
            **kwargs,
        )

        self.cone = Cone(
            direction=self.direction, base_radius=base_radius, height=height, **kwargs
        )
        self.cone.shift(end)
        self.add(self.cone)
        self.set_color(color)


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
        u_range: Sequence[float] = (0, TAU),
        v_range: Sequence[float] = (0, TAU),
        resolution: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        if config.renderer == RendererType.OPENGL:
            res_value = (101, 101)
        elif config.renderer == RendererType.CAIRO:
            res_value = (24, 24)

        resolution = resolution if resolution is not None else res_value

        self.R = major_radius
        self.r = minor_radius
        super().__init__(
            self.func,
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,
            **kwargs,
        )

    def func(self, u: float, v: float) -> np.ndarray:
        """The z values defining the :class:`Torus` being plotted.

        Returns
        -------
        :class:`numpy.ndarray`
            The z values defining the :class:`Torus`.
        """
        P = np.array([np.cos(u), np.sin(u), 0])
        return (self.R - self.r * np.cos(v)) * P - self.r * np.sin(v) * OUT
