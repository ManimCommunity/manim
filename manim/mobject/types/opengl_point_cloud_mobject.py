__all__ = ["OpenGLPMobject", "OpenGLPGroup", "OpenGLPMPoint"]

import moderngl
import numpy as np

from ...constants import *
from ...mobject.opengl_mobject import OpenGLMobject
from ...utils.bezier import interpolate
from ...utils.color import BLACK, WHITE, YELLOW, color_gradient, color_to_rgba
from ...utils.config_ops import _Uniforms
from ...utils.iterables import resize_with_interpolation


class OpenGLPMobject(OpenGLMobject):
    shader_folder = "true_dot"
    # Scale for consistency with cairo units
    OPENGL_POINT_RADIUS_SCALE_FACTOR = 0.01
    shader_dtype = [
        ("point", np.float32, (3,)),
        ("color", np.float32, (4,)),
    ]

    point_radius = _Uniforms()

    def __init__(
        self, stroke_width=2.0, color=YELLOW, render_primitive=moderngl.POINTS, **kwargs
    ):
        self.stroke_width = stroke_width
        super().__init__(color=color, render_primitive=render_primitive, **kwargs)
        self.point_radius = (
            self.stroke_width * OpenGLPMobject.OPENGL_POINT_RADIUS_SCALE_FACTOR
        )

    def reset_points(self):
        self.rgbas = np.zeros((1, 4))
        self.points = np.zeros((0, 3))
        return self

    def get_array_attrs(self):
        return ["points", "rgbas"]

    def add_points(self, points, rgbas=None, color=None, opacity=None):
        """Add points.

        Points must be a Nx3 numpy array.
        Rgbas must be a Nx4 numpy array if it is not None.
        """
        if rgbas is None and color is None:
            color = YELLOW
        self.append_points(points)
        # rgbas array will have been resized with points
        if color is not None:
            if opacity is None:
                opacity = self.rgbas[-1, 3]
            new_rgbas = np.repeat([color_to_rgba(color, opacity)], len(points), axis=0)
        elif rgbas is not None:
            new_rgbas = rgbas
        elif len(rgbas) != len(points):
            raise ValueError("points and rgbas must have same length")
        self.rgbas = np.append(self.rgbas, new_rgbas, axis=0)
        return self

    def thin_out(self, factor=5):
        """
        Removes all but every nth point for n = factor
        """
        for mob in self.family_members_with_points():
            num_points = mob.get_num_points()

            def thin_func():
                return np.arange(0, num_points, factor)

            if len(mob.points) == len(mob.rgbas):
                mob.set_rgba_array_direct(mob.rgbas[thin_func()])
            mob.set_points(mob.points[thin_func()])

        return self

    def set_color_by_gradient(self, *colors):
        self.rgbas = np.array(
            list(map(color_to_rgba, color_gradient(*colors, self.get_num_points()))),
        )
        return self

    def set_colors_by_radial_gradient(
        self,
        center=None,
        radius=1,
        inner_color=WHITE,
        outer_color=BLACK,
    ):
        start_rgba, end_rgba = list(map(color_to_rgba, [inner_color, outer_color]))
        if center is None:
            center = self.get_center()
        for mob in self.family_members_with_points():
            distances = np.abs(self.points - center)
            alphas = np.linalg.norm(distances, axis=1) / radius

            mob.rgbas = np.array(
                np.array(
                    [interpolate(start_rgba, end_rgba, alpha) for alpha in alphas],
                ),
            )
        return self

    def match_colors(self, pmobject):
        self.rgbas[:] = resize_with_interpolation(pmobject.rgbas, self.get_num_points())
        return self

    def fade_to(self, color, alpha, family=True):
        rgbas = interpolate(self.rgbas, color_to_rgba(color), alpha)
        for mob in self.submobjects:
            mob.fade_to(color, alpha, family)
        self.set_rgba_array_direct(rgbas)
        return self

    def filter_out(self, condition):
        for mob in self.family_members_with_points():
            to_keep = ~np.apply_along_axis(condition, 1, mob.points)
            for key in mob.data:
                mob.data[key] = mob.data[key][to_keep]
        return self

    def sort_points(self, function=lambda p: p[0]):
        """
        function is any map from R^3 to R
        """
        for mob in self.family_members_with_points():
            indices = np.argsort(np.apply_along_axis(function, 1, mob.points))
            for key in mob.data:
                mob.data[key] = mob.data[key][indices]
        return self

    def ingest_submobjects(self):
        for key in self.data:
            self.data[key] = np.vstack([sm.data[key] for sm in self.get_family()])
        return self

    def point_from_proportion(self, alpha):
        index = alpha * (self.get_num_points() - 1)
        return self.points[int(index)]

    def pointwise_become_partial(self, pmobject, a, b):
        lower_index = int(a * pmobject.get_num_points())
        upper_index = int(b * pmobject.get_num_points())
        for key in self.data:
            self.data[key] = pmobject.data[key][lower_index:upper_index]
        return self

    def get_shader_data(self):
        shader_data = np.zeros(len(self.points), dtype=self.shader_dtype)
        self.read_data_to_shader(shader_data, "point", "points")
        self.read_data_to_shader(shader_data, "color", "rgbas")
        return shader_data


class OpenGLPGroup(OpenGLPMobject):
    def __init__(self, *pmobs, **kwargs):
        if not all([isinstance(m, OpenGLPMobject) for m in pmobs]):
            raise Exception("All submobjects must be of type OpenglPMObject")
        super().__init__(**kwargs)
        self.add(*pmobs)

    def fade_to(self, color, alpha, family=True):
        if family:
            for mob in self.submobjects:
                mob.fade_to(color, alpha, family)


class OpenGLPMPoint(OpenGLPMobject):
    def __init__(self, location=ORIGIN, stroke_width=4.0, **kwargs):
        self.location = location
        super().__init__(stroke_width=stroke_width, **kwargs)

    def init_points(self):
        self.points = np.array([self.location], dtype=np.float32)
