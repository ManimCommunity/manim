"""Three-dimensional mobjects."""

__all__ = [
    "ThreeDVMobject",
    "Surface",
    "ParametricSurface",
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


from typing import *

import numpy as np
from colour import Color

from manim.mobject.opengl_compatibility import ConvertToOpenGL

from .. import config
from ..constants import *
from ..mobject.geometry import Circle, Square
from ..mobject.mobject import *
from ..mobject.opengl_mobject import OpenGLMobject
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import *
from ..utils.deprecation import deprecated, deprecated_params
from ..utils.iterables import tuplify
from ..utils.space_ops import normalize, perpendicular_bisector, z_to_vector


class ThreeDVMobject(VMobject, metaclass=ConvertToOpenGL):
    def __init__(self, shade_in_3d=True, **kwargs):
        super().__init__(shade_in_3d=shade_in_3d, **kwargs)


class Surface(VGroup, metaclass=ConvertToOpenGL):
    """Creates a Parametric Surface using a checkerboard pattern.

    Parameters
    ----------
    func :
        The function that defines the surface.
    u_range :
        The range of the ``u`` variable: ``(u_min, u_max)``.
    v_range :
        The range of the ``v`` variable: ``(v_min, v_max)``.
    resolution :
        The number of samples taken of the surface. A tuple
        can be used to define different resolutions for ``u`` and
        ``v`` respectively.

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
                    v_range=[0, TAU]
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
        fill_color: "Color" = BLUE_D,
        fill_opacity: float = 1.0,
        checkerboard_colors: Sequence["Color"] = [BLUE_D, BLUE_E],
        stroke_color: "Color" = LIGHT_GREY,
        stroke_width: float = 0.5,
        should_make_jagged: bool = False,
        pre_function_handle_to_anchor_scale_factor: float = 0.00001,
        **kwargs
    ) -> None:
        self.u_range = u_range
        self.v_range = v_range
        super().__init__(**kwargs)
        self.resolution = resolution
        self.surface_piece_config = surface_piece_config
        self.fill_color = fill_color
        self.fill_opacity = fill_opacity
        self.checkerboard_colors = checkerboard_colors
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.should_make_jagged = should_make_jagged
        self.pre_function_handle_to_anchor_scale_factor = (
            pre_function_handle_to_anchor_scale_factor
        )
        self.func = func
        self.setup_in_uv_space()
        self.apply_function(lambda p: func(p[0], p[1]))
        if self.should_make_jagged:
            self.make_jagged()

    def get_u_values_and_v_values(self):
        res = tuplify(self.resolution)
        if len(res) == 1:
            u_res = v_res = res[0]
        else:
            u_res, v_res = res

        u_values = np.linspace(*self.u_range, u_res + 1)
        v_values = np.linspace(*self.v_range, v_res + 1)

        return u_values, v_values

    def setup_in_uv_space(self):
        u_values, v_values = self.get_u_values_and_v_values()
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

    def set_fill_by_checkerboard(self, *colors, opacity=None):
        n_colors = len(colors)
        for face in self:
            c_index = (face.u_index + face.v_index) % n_colors
            face.set_fill(colors[c_index], opacity=opacity)
        return self

    def set_fill_by_value(self, axes: "Mobject", colors: Union[Iterable[Color], Color]):
        """Sets the color of each mobject of a parametric surface to a color relative to its z-value

        Parameters
        ----------
        axes :
            The axes for the parametric surface, which will be used to map z-values to colors.
        colors :
            A list of colors, ordered from lower z-values to higher z-values. If a list of tuples is passed
            containing colors paired with numbers, then those numbers will be used as the pivots.

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
                    resolution_fa = 42
                    self.set_camera_orientation(phi=75 * DEGREES, theta=-120 * DEGREES)
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
                    surface_plane.set_fill_by_value(axes=axes, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)])
                    self.add(axes, surface_plane)
        """
        if type(colors[0]) is tuple:
            new_colors, pivots = [[i for i, j in colors], [j for i, j in colors]]
        else:
            new_colors = colors

            pivot_min = axes.z_range[0]
            pivot_max = axes.z_range[1]
            pivot_frequency = (pivot_max - pivot_min) / (len(new_colors) - 1)
            pivots = np.arange(
                start=pivot_min,
                stop=pivot_max + pivot_frequency,
                step=pivot_frequency,
            )

        for mob in self.family_members_with_points():
            z_value = axes.point_to_coords(mob.get_midpoint())[2]
            if z_value <= pivots[0]:
                mob.set_color(new_colors[0])
            elif z_value >= pivots[-1]:
                mob.set_color(new_colors[-1])
            else:
                for i, pivot in enumerate(pivots):
                    if pivot > z_value:
                        color_index = (z_value - pivots[i - 1]) / (
                            pivots[i] - pivots[i - 1]
                        )
                        color_index = min(color_index, 1)
                        mob_color = interpolate_color(
                            new_colors[i - 1],
                            new_colors[i],
                            color_index,
                        )
                        if config.renderer == "opengl":
                            mob.set_color(mob_color, recurse=False)
                        else:
                            mob.set_color(mob_color, family=False)
                        break

        return self


@deprecated(since="v0.10.0", replacement=Surface)
class ParametricSurface(Surface):
    # shifts inheritance from Surface/OpenGLSurface depending on the renderer.
    """Creates a parametric surface"""


# Specific shapes


class Sphere(Surface):
    """A mobject representing a three-dimensional sphere.

    Examples
    ---------

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
        center=ORIGIN,
        radius=1,
        resolution=None,
        u_range=(0, TAU),
        v_range=(0, PI),
        **kwargs
    ):
        if config.renderer == "opengl":
            res_value = (101, 51)
        else:
            res_value = (24, 12)

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

    def func(self, u, v):
        return self.radius * np.array(
            [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)],
        )


class Dot3D(Sphere):
    """A spherical dot.

    Parameters
    --------
    point : Union[:class:`list`, :class:`numpy.ndarray`], optional
        The location of the dot.
    radius : :class:`float`, optional
        The radius of the dot.
    color : :class:`~.Colors`, optional
        The color of the :class:`Dot3D`

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
        point=ORIGIN,
        radius=DEFAULT_DOT_RADIUS,
        color=WHITE,
        resolution=(8, 8),
        **kwargs
    ):
        super().__init__(center=point, radius=radius, resolution=resolution, **kwargs)
        self.set_color(color)


class Cube(VGroup):
    def __init__(
        self,
        side_length=2,
        fill_opacity=0.75,
        fill_color=BLUE,
        stroke_width=0,
        **kwargs
    ):
        self.side_length = side_length
        super().__init__(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs,
        )

    def generate_points(self):
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
    """A cuboid.

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

    def __init__(self, dimensions=[3, 2, 1], **kwargs):
        self.dimensions = dimensions
        super().__init__(**kwargs)

    def generate_points(self):
        super().generate_points()
        for dim, value in enumerate(self.dimensions):
            self.rescale_to_fit(value, dim, stretch=True)


class Cone(Surface):
    """A circular cone.
    Can be defined using 2 parameters: its height, and its base radius.
    The polar angle, theta, can be calculated using arctan(base_radius /
    height) The spherical radius, r, is calculated using the pythagorean
    theorem.

    Examples
    --------
    .. manim:: ExampleCone
        :save_last_frame:

        class ExampleCone(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                cone = Cone(direction=X_AXIS+Y_AXIS+2*Z_AXIS)
                self.set_camera_orientation(phi=5*PI/11, theta=PI/9)
                self.add(axes, cone)

    Parameters
    --------
    base_radius : :class:`float`
        The base radius from which the cone tapers.
    height : :class:`float`
        The height measured from the plane formed by the base_radius to the apex of the cone.
    direction : :class:`numpy.array`
        The direction of the apex.
    show_base : :class:`bool`
        Whether to show the base plane or not.
    v_range : :class:`Sequence[float]`
        The azimuthal angle to start and end at.
    u_min : :class:`float`
        The radius at the apex.
    checkerboard_colors : :class:`bool`
        Show checkerboard grid texture on the cone.
    """

    def __init__(
        self,
        base_radius=1,
        height=1,
        direction=Z_AXIS,
        show_base=False,
        v_range=[0, TAU],
        u_min=0,
        checkerboard_colors=False,
        **kwargs
    ):
        self.direction = direction
        self.theta = PI - np.arctan(base_radius / height)

        super().__init__(
            self.func,
            v_range=v_range,
            u_range=[u_min, np.sqrt(base_radius ** 2 + height ** 2)],
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

    def func(self, u, v):
        """Converts from spherical coordinates to cartesian.
        Parameters
        ---------
        u : :class:`float`
            The radius.
        v : :class:`float`
            The azimuthal angle.
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

    def _rotate_to_direction(self):
        x, y, z = self.direction

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)

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

    def set_direction(self, direction):
        self.direction = direction
        self._rotate_to_direction()

    def get_direction(self):
        return self.direction


class Cylinder(Surface):
    """A cylinder, defined by its height, radius and direction,

    Examples
    ---------
    .. manim:: ExampleCylinder
        :save_last_frame:

        class ExampleCylinder(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                cylinder = Cylinder(radius=2, height=3)
                self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                self.add(axes, cylinder)

    Parameters
    ---------
    radius : :class:`float`
        The radius of the cylinder.
    height : :class:`float`
        The height of the cylinder.
    direction : :class:`numpy.array`
        The direction of the central axis of the cylinder.
    v_range : :class:`Sequence[float]`
        The height along the height axis (given by direction) to start and end on.
    show_ends : :class:`bool`
        Whether to show the end caps or not.
    """

    def __init__(
        self,
        radius=1,
        height=2,
        direction=Z_AXIS,
        v_range=[0, TAU],
        show_ends=True,
        resolution=(24, 24),
        **kwargs
    ):
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

    def func(self, u, v):
        """Converts from cylindrical coordinates to cartesian.
        Parameters
        ---------
        u : :class:`float`
            The height.
        v : :class:`float`
            The azimuthal angle.
        """
        height = u
        phi = v
        r = self.radius
        return np.array([r * np.cos(phi), r * np.sin(phi), height])

    def add_bases(self):
        """Adds the end caps of the cylinder."""
        color = self.color if config["renderer"] == "opengl" else self.fill_color
        opacity = self.opacity if config["renderer"] == "opengl" else self.fill_opacity
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

    def _rotate_to_direction(self):
        x, y, z = self.direction

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)

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

    def set_direction(self, direction):
        # if get_norm(direction) is get_norm(self.direction):
        #     pass
        self.direction = direction
        self._rotate_to_direction()

    def get_direction(self):
        return self.direction


class Line3D(Cylinder):
    """A cylindrical line, for use in ThreeDScene.

    Examples
    ---------
    .. manim:: ExampleLine3D
        :save_last_frame:

        class ExampleLine3D(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                line = Line3D(start=np.array([0, 0, 0]), end=np.array([2, 2, 2]))
                self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                self.add(axes, line)

    Parameters
    ---------
    start : :class:`numpy.array`
        The start position of the line.
    end : :class:`numpy.array`
        The end position of the line.
    thickness : :class:`float`
        The thickness of the line.
    """

    def __init__(self, start=LEFT, end=RIGHT, thickness=0.02, color=None, **kwargs):
        self.thickness = thickness
        self.set_start_and_end_attrs(start, end, **kwargs)
        if color is not None:
            self.set_color(color)

    def set_start_and_end_attrs(self, start, end, **kwargs):
        """Sets the start and end points of the line.

        If either ``start`` or ``end`` are :class:`Mobjects <.Mobject>`, this gives their centers.
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

    def pointify(self, mob_or_point, direction=None):
        if isinstance(mob_or_point, (Mobject, OpenGLMobject)):
            mob = mob_or_point
            if direction is None:
                return mob.get_center()
            else:
                return mob.get_boundary_point(direction)
        return np.array(mob_or_point)

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    @classmethod
    def parallel_to(
        cls,
        line: "Line3D",
        point: Sequence[float] = ORIGIN,
        length: float = 5,
        **kwargs
    ):
        """Returns a line parallel to another line going through
        a given point.

        Parameters
        ----------
        line
            The line to be parallel to.
        point
            The point to pass through.
        kwargs
            Additional parameters to be passed to the class.

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
        line: "Line3D",
        point: Sequence[float] = ORIGIN,
        length: float = 5,
        **kwargs
    ):
        """Returns a line perpendicular to another line going through
        a given point.

        Parameters
        ----------
        line
            The line to be perpendicular to.
        point
            The point to pass through.
        kwargs
            Additional parameters to be passed to the class.

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

    Examples
    ---------
    .. manim:: ExampleArrow3D
        :save_last_frame:

        class ExampleArrow3D(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                arrow = Arrow3D(start=np.array([0, 0, 0]), end=np.array([2, 2, 2]))
                self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                self.add(axes, arrow)

    Parameters
    ---------
    start : :class:`numpy.array`
        The start position of the arrow.
    end : :class:`numpy.array`
        The end position of the arrow.
    thickness : :class:`float`
        The thickness of the arrow.
    height : :class:`float`
        The height of the conical tip.
    base_radius: :class:`float`
        The base radius of the conical tip.
    """

    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        thickness=0.02,
        height=0.3,
        base_radius=0.08,
        color=WHITE,
        **kwargs
    ):
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

    Examples
    ---------
    .. manim :: ExampleTorus
        :save_last_frame:

        class ExampleTorus(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                torus = Torus()
                self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
                self.add(axes, torus)

    Parameters
    ---------
    major_radius : :class:`float`
        Distance from the center of the tube to the center of the torus.
    minor_radius : :class:`float`
        Radius of the tube.
    """

    def __init__(
        self,
        major_radius=3,
        minor_radius=1,
        u_range=(0, TAU),
        v_range=(0, TAU),
        resolution=None,
        **kwargs
    ):
        if config.renderer == "opengl":
            res_value = (101, 101)
        else:
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

    def func(self, u, v):
        P = np.array([np.cos(u), np.sin(u), 0])
        return (self.R - self.r * np.cos(v)) * P - self.r * np.sin(v) * OUT
