import math

import numpy as np

from ..constants import *
from ..mobject.types.opengl_surface import OpenGLSurface
from ..mobject.types.opengl_vectorized_mobject import OpenGLVGroup, OpenGLVMobject


class OpenGLSurfaceMesh(OpenGLVGroup):
    def __init__(
        self,
        uv_surface,
        resolution=None,
        stroke_width=1,
        normal_nudge=1e-2,
        depth_test=True,
        flat_stroke=False,
        **kwargs
    ):
        if not isinstance(uv_surface, OpenGLSurface):
            raise Exception("uv_surface must be of type OpenGLSurface")
        self.uv_surface = uv_surface
        self.resolution = resolution if resolution is not None else (21, 21)
        self.normal_nudge = normal_nudge
        super().__init__(
            stroke_width=stroke_width,
            depth_test=depth_test,
            flat_stroke=flat_stroke,
            **kwargs
        )

    def init_points(self):
        uv_surface = self.uv_surface

        full_nu, full_nv = uv_surface.resolution
        part_nu, part_nv = self.resolution
        u_indices = np.linspace(0, full_nu, part_nu).astype(int)
        v_indices = np.linspace(0, full_nv, part_nv).astype(int)

        points, du_points, dv_points = uv_surface.get_surface_points_and_nudged_points()
        normals = uv_surface.get_unit_normals()
        nudged_points = points + self.normal_nudge * normals

        for ui in u_indices:
            path = OpenGLVMobject()
            full_ui = full_nv * ui
            path.set_points_smoothly(nudged_points[full_ui : full_ui + full_nv])
            self.add(path)
        for vi in v_indices:
            path = OpenGLVMobject()
            path.set_points_smoothly(nudged_points[vi::full_nv])
            self.add(path)


class OpenGLSphere(OpenGLSurface):
    def __init__(self, resolution=None, radius=1, u_range=None, v_range=None, **kwargs):
        resolution = resolution if resolution is not None else (101, 51)
        u_range = u_range if u_range is not None else (0, TAU)
        v_range = v_range if v_range is not None else (0, PI)
        self.radius = radius
        super().__init__(
            resolution=resolution, u_range=u_range, v_range=v_range, **kwargs
        )

    def uv_func(self, u, v):
        return self.radius * np.array(
            [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)]
        )


class OpenGLTorus(OpenGLSurface):
    def __init__(self, u_range=None, v_range=None, r1=3, r2=1, **kwargs):
        u_range = u_range if u_range is not None else (0, TAU)
        v_range = v_range if v_range is not None else (0, TAU)
        self.r1 = r1
        self.r2 = r2
        super().__init__(u_range=u_range, v_range=v_range, **kwargs)

    def uv_func(self, u, v):
        P = np.array([math.cos(u), math.sin(u), 0])
        return (self.r1 - self.r2 * math.cos(v)) * P - math.sin(v) * OUT
