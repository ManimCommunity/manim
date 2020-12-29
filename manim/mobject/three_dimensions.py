"""Three-dimensional mobjects."""

__all__ = ["ThreeDVMobject", "ParametricSurface", "Sphere", "Cube", "Prism"]

from ..constants import *
from ..mobject.geometry import Square
from ..mobject.types.vectorized_mobject import VGroup
from ..mobject.types.vectorized_mobject import VMobject
from ..utils.iterables import tuplify
from ..utils.space_ops import z_to_vector
from ..utils.color import BLUE_D, BLUE, BLUE_E, LIGHT_GREY

##############


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
            **kwargs
        )
        self.radius = radius
        self.scale(self.radius)

    def func(
        self, u, v
    ):  # FIXME: An attribute defined in manim.mobject.three_dimensions line 56 hides this method
        return np.array([np.cos(v) * np.sin(u), np.sin(v) * np.sin(u), np.cos(u)])


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
            **kwargs
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
