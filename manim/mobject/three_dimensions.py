"""Three-dimensional mobjects."""

__all__ = [
    "ThreeDVMobject",
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

import numpy as np

from ..constants import *
from ..mobject.geometry import Circle, Square
from ..mobject.mobject import *
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.color import *
from ..utils.iterables import tuplify
from ..utils.space_ops import normalize, z_to_vector


class ThreeDVMobject(VMobject):
    def __init__(self, shade_in_3d=True, **kwargs):
        super().__init__(shade_in_3d=shade_in_3d, **kwargs)


class ParametricSurface(VGroup):
    def __init__(
        self,
        func,
        u_min=0,
        u_max=1,
        v_min=0,
        v_max=1,
        resolution=32,
        surface_piece_config={},
        fill_color=BLUE_D,
        fill_opacity=1.0,
        checkerboard_colors=[BLUE_D, BLUE_E],
        stroke_color=LIGHT_GREY,
        stroke_width=0.5,
        should_make_jagged=False,
        pre_function_handle_to_anchor_scale_factor=0.00001,
        **kwargs
    ):
        VGroup.__init__(self, **kwargs)
        self.u_min = u_min
        self.u_max = u_max
        self.v_min = v_min
        self.v_max = v_max
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
        u_min = self.u_min
        u_max = self.u_max
        v_min = self.v_min
        v_max = self.v_max

        u_values = np.linspace(u_min, u_max, u_res + 1)
        v_values = np.linspace(v_min, v_max, v_res + 1)

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
                    ]
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


# Specific shapes


class Sphere(ParametricSurface):
    def __init__(
        self,
        center=ORIGIN,
        radius=1,
        resolution=(12, 24),
        u_min=0.001,
        u_max=PI - 0.001,
        v_min=0,
        v_max=TAU,
        **kwargs
    ):
        ParametricSurface.__init__(
            self,
            self.func,
            resolution=resolution,
            u_min=u_min,
            u_max=u_max,
            v_min=v_min,
            v_max=v_max,
            **kwargs,
        )
        self.radius = radius
        self.scale(self.radius)
        self.shift(center)

    def func(
        self, u, v
    ):  # FIXME: An attribute defined in manim.mobject.three_dimensions line 56 hides this method
        return np.array([np.cos(v) * np.sin(u), np.sin(v) * np.sin(u), np.cos(u)])


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

    def __init__(self, point=ORIGIN, radius=DEFAULT_DOT_RADIUS, color=WHITE, **kwargs):
        Sphere.__init__(self, center=point, radius=radius, **kwargs)
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


class Prism(Cube):
    def __init__(self, dimensions=[3, 2, 1], **kwargs):
        self.dimensions = dimensions
        Cube.__init__(self, **kwargs)

    def generate_points(self):
        Cube.generate_points(self)
        for dim, value in enumerate(self.dimensions):
            self.rescale_to_fit(value, dim, stretch=True)


class Cone(ParametricSurface):
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
    v_min : :class:`float`
        The azimuthal angle to start at.
    v_max : :class:`float`
        The azimuthal angle to end at.
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
        v_min=0,
        v_max=TAU,
        u_min=0,
        checkerboard_colors=False,
        **kwargs
    ):
        self.direction = direction
        self.theta = PI - np.arctan(base_radius / height)

        ParametricSurface.__init__(
            self,
            self.func,
            v_min=v_min,
            v_max=v_max,
            u_min=u_min,
            u_max=np.sqrt(base_radius ** 2 + height ** 2),
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
            ]
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


class Cylinder(ParametricSurface):
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
    v_min : :class:`float`
        The height along the height axis (given by direction) to start on.
    v_max : :class:`float`
        The height along the height axis (given by direction) to end on.
    show_ends : :class:`bool`
        Whether to show the end caps or not.
    """

    def __init__(
        self,
        radius=1,
        height=2,
        direction=Z_AXIS,
        v_min=0,
        v_max=TAU,
        show_ends=True,
        resolution=24,
        **kwargs
    ):
        self._height = height
        self.radius = radius
        ParametricSurface.__init__(
            self,
            self.func,
            resolution=resolution,
            u_min=-self._height / 2,
            u_max=self._height / 2,
            v_min=v_min,
            v_max=v_max,
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
        self.base_top = Circle(
            radius=self.radius,
            color=self.fill_color,
            fill_opacity=self.fill_opacity,
            shade_in_3d=True,
            stroke_width=0,
        )
        self.base_top.shift(self.u_max * IN)
        self.base_bottom = Circle(
            radius=self.radius,
            color=self.fill_color,
            fill_opacity=self.fill_opacity,
            shade_in_3d=True,
            stroke_width=0,
        )
        self.base_bottom.shift(self.u_min * IN)
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
        Cylinder.__init__(
            self,
            height=np.linalg.norm(self.vect),
            radius=self.thickness,
            direction=self.direction,
            **kwargs,
        )
        self.shift((self.start + self.end) / 2)

    def pointify(self, mob_or_point, direction=None):
        if isinstance(mob_or_point, Mobject):
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
        height=0.5,
        base_radius=0.25,
        color=WHITE,
        **kwargs
    ):
        Line3D.__init__(self, start=start, end=end, **kwargs)

        self.length = np.linalg.norm(self.vect)
        self.set_start_and_end_attrs(
            self.start,
            self.end - height * self.direction,
            thickness=thickness,
            **kwargs,
        )

        self.cone = Cone(
            direction=self.direction, base_radius=base_radius, height=height, **kwargs
        )
        self.cone.shift(end)
        self.add(self.cone)
        self.set_color(color)


class Torus(ParametricSurface):
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
        u_min=0,
        u_max=TAU,
        v_min=0,
        v_max=TAU,
        resolution=24,
        **kwargs
    ):
        self.R = major_radius
        self.r = minor_radius
        ParametricSurface.__init__(
            self,
            self.func,
            u_min=u_min,
            u_max=u_max,
            v_min=v_min,
            v_max=v_max,
            resolution=resolution,
            **kwargs,
        )

    def func(self, u, v):
        P = np.array([np.cos(u), np.sin(u), 0])
        return (self.R - self.r * np.cos(v)) * P - self.r * np.sin(v) * OUT
