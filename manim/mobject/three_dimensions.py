"""Three-dimensional mobjects."""

__all__ = ["ThreeDVMobject", "ParametricSurface", "Sphere", "Cube", "Prism"]

from ..constants import *
from ..mobject.geometry import Square, Dot, Line
from ..mobject.types.vectorized_mobject import VGroup
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.iterables import tuplify
from ..utils.space_ops import z_to_vector
from ..utils.color import BLUE_D, BLUE, BLUE_E, LIGHT_GREY

##############


class ThreeDVMobject(VMobject):
    CONFIG = {
        "shade_in_3d": True,
    }


class ParametricSurface(VGroup):
    CONFIG = {
        "u_min": 0,
        "u_max": 1,
        "v_min": 0,
        "v_max": 1,
        "resolution": 32,
        "surface_piece_config": {},
        "fill_color": BLUE_D,
        "fill_opacity": 1.0,
        "checkerboard_colors": [BLUE_D, BLUE_E],
        "stroke_color": LIGHT_GREY,
        "stroke_width": 0.5,
        "should_make_jagged": False,
        "pre_function_handle_to_anchor_scale_factor": 0.00001,
    }

    def __init__(self, func, **kwargs):
        VGroup.__init__(self, **kwargs)
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


# Specific shapes


class Sphere(ParametricSurface):
    CONFIG = {
        "resolution": (12, 24),
        "radius": 1,
        "u_min": 0.001,
        "u_max": PI - 0.001,
        "v_min": 0,
        "v_max": TAU,
    }

    def __init__(self, **kwargs):
        ParametricSurface.__init__(self, self.func, **kwargs)
        self.scale(self.radius)

    def func(self, u, v):
        return np.array([np.cos(v) * np.sin(u), np.sin(v) * np.sin(u), np.cos(u)])


class Cone(ParametricSurface):
    CONFIG = {
        "base_radius": 1,
        "height": 1,
        "direction": Z_AXIS,
        "show_base": False,

        "resolution": 24,
        # v will stand for phi
        "v_min": 0,
        "v_max": TAU,
        # u will stand for r
        "u_min": 0,
        # u_max is calculated as a property

        "checkerboard_colors": False,
    }
    """
               |\\
               |_\\ <-- theta
    height --> |  \\
               |   \\ <-- r
               |    \\
               |     \\
               --------
               base_radius
    """

    def __init__(self, **kwargs):
        ParametricSurface.__init__(
            self, self.func, **kwargs
        )
        # used for rotations
        self._current_theta = 0
        self._current_phi = 0

        if self.show_base:
            self.base_circle = Dot(
                point=self.height*IN,
                radius=self.base_radius,
                color=self.fill_color,
                fill_opacity=self.fill_opacity,
            )
            self.add(self.base_circle)

        self._rotate_to_direction()


    @property
    def u_max(self):
        return np.sqrt(self.base_radius**2 + self.height**2)
    @property
    def theta(self):
        return PI - np.arctan(self.base_radius / self.height)

    def func(self, u, v):
        """
        u is r
        v is phi
        theta is available from self.theta
        """
        r = u
        phi = v
        theta = self.theta
        return np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])

    def _rotate_to_direction(self):
        x, y, z = self.direction

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)

        if x == 0:
            if y == 0: # along the z axis
                phi = 0
            else:
                phi = np.arctan(np.inf)
                if y < 0:
                    phi += PI
        else:
            phi = np.arctan(y/x)
        if x < 0:
            phi += PI

        # this is a lazy implementation of rotations
        # This un-rotates the last rotation (which is stored im memory)
        # and then calculates a rotation against the z axis
        # A better way would be to get the angle between the current direction
        # and the new one (using the dot product for example) and rotate it about
        # a 3rd vector, normal to the previous two

        # undo old rotation (in reverse order)
        self.rotate(-self._current_phi  , Z_AXIS, about_point=ORIGIN)
        self.rotate(-self._current_theta, Y_AXIS, about_point=ORIGIN)
        # do new rotation
        self.rotate(theta, Y_AXIS, about_point=ORIGIN)
        self.rotate(phi  , Z_AXIS, about_point=ORIGIN)
        # store values
        self._current_theta = theta
        self._current_phi = phi

    def set_direction(self, direction):
        self.direction = direction
        self._rotate_to_direction()

    def get_direction(self):
        return self._current_theta, self._current_phi

class Arrow3d(VGroup):
    CONFIG = {
        "cone_config": {
            "height": .5,
            "base_radius": .25,
        },
        "color": WHITE,
    }

    def __init__(self, start=LEFT, end=RIGHT, **kwargs):
        VGroup.__init__(self, **kwargs)

        # TODO: inherit color
        self.line = Line(start, end)
        self.line.set_color(self.color)

        self.cone = Cone(
            direction=self.direction,
            **self.cone_config
        )
        self.cone.shift(self.end)
        self.cone.set_color(self.color)

        self.add(self.line, self.cone)

    @property
    def start(self):
        return self.line.start
    @property
    def end(self):
        return self.line.end
    @property
    def direction(self):
        return self.end - self.start


class Cube(VGroup):
    CONFIG = {
        "fill_opacity": 0.75,
        "fill_color": BLUE,
        "stroke_width": 0,
        "side_length": 2,
    }

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
    CONFIG = {"dimensions": [3, 2, 1]}

    def generate_points(self):
        Cube.generate_points(self)
        for dim, value in enumerate(self.dimensions):
            self.rescale_to_fit(value, dim, stretch=True)


class Cylinder(ParametricSurface):
    CONFIG = {
        "resolution": 24,

        "radius": 1,
        "height": 2,
        "direction": Z_AXIS,

        "center_point": ORIGIN,

        # v will is the polar angle
        "v_min": 0,
        "v_max": TAU,
    }
    # u is the height
    @property
    def u_min(self):
        return - self.height / 2
    @property
    def u_max(self):
        return   self.height / 2

    def __init__(self, **kwargs):
        ParametricSurface.__init__(
            self, self.func, **kwargs
        )
        self.add_bases()
        self._rotate_to_direction()

    def func(self, u, v):
        """
        u is height
        v is phi
        """
        height = u
        phi = v
        r = self.radius
        return np.array([
            r * np.cos(phi),
            r * np.sin(phi),
            height
        ])

    def add_bases(self):
        self.base_top = Dot(
            point=self.u_max*IN,
            radius=self.radius,
            color=self.fill_color,
            fill_opacity=self.fill_opacity,
        )
        self.base_bottom = Dot(
            point=self.u_min*IN,
            radius=self.radius,
            color=self.fill_color,
            fill_opacity=self.fill_opacity,
        )
        self.add(self.base_top, self.base_bottom)

    def _rotate_to_direction(self):
        x, y, z = self.direction

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/r)

        if x == 0:
            if y == 0: # along the z axis
                phi = 0
            else:
                phi = np.arctan(np.inf)
                if y < 0:
                    phi += PI
        else:
            phi = np.arctan(y/x)
        if x < 0:
            phi += PI

        self.rotate(theta, Y_AXIS, about_point=ORIGIN)
        self.rotate(phi  , Z_AXIS, about_point=ORIGIN)

    def set_direction(self, direction):
        self.direction = direction
        self._rotate_to_direction()


class Paraboloid(ParametricSurface):
    CONFIG = {
        "resolution": 24,

        # z = x_factor * x^2 + y_factor * y^2
        "x_factor": 1,
        "y_factor": 1,

        "center_point": ORIGIN,
    }
    @property
    def u_min(self):
        raise NotImplemented
    @property
    def u_max(self):
        raise NotImplemented
    @property
    def v_min(self):
        raise NotImplemented
    @property
    def v_max(self):
        raise NotImplemented

    def __init__(self, **kwargs):
        ParametricSurface.__init__(
            self, self.func, **kwargs
        )
        self.shift(self.center_point)

    def func(self, u, v):
        raise NotImplemented

    def _paraboloid(self, x, y):
        return self.x_factor * x**2 + self.y_factor * y**2

    def get_value_at_point(self, point):
        # point may be 2d or 3d array
        x = point[0] - self.center_point[0]
        y = point[1] - self.center_point[1]
        return self._paraboloid(x, y)

    def get_gradient(self, point):
        # point may be 2d or 3d array
        x = point[0] - self.center_point[0]
        y = point[1] - self.center_point[1]
        return np.array([
            2*self.x_factor*x,
            2*self.y_factor*y,
            0
        ])

class ParaboloidCartesian(Paraboloid):
    CONFIG = {
        "x_min": -2,
        "x_max": 2,
        "y_min": -2,
        "y_max": 2,
    }
    @property
    def u_min(self):
        return self.x_min
    @property
    def u_max(self):
        return self.x_max
    @property
    def v_min(self):
        return self.y_min
    @property
    def v_max(self):
        return self.y_max

    def func(self, x, y):
        return np.array([
            x,
            y,
            self.x_factor * x**2 + self.y_factor * y**2
        ])
class ParaboloidPolar(Paraboloid):
    CONFIG = {
        "r_min": 0,
        "r_max": 2,
        "theta_min": 0,
        "theta_max": TAU,
    }
    @property
    def u_min(self):
        return self.r_min
    @property
    def u_max(self):
        return self.r_max
    @property
    def v_min(self):
        return self.theta_min
    @property
    def v_max(self):
        return self.theta_max

    def func(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.array([
            x,
            y,
            self.x_factor * x**2 + self.y_factor * y**2
        ])
