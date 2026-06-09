from __future__ import annotations

from typing import Any

import numpy as np

from manim.mobject.opengl.opengl_surface import OpenGLSurface
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVGroup, OpenGLVMobject

__all__ = ["OpenGLSurfaceMesh"]


class OpenGLSurfaceMesh(OpenGLVGroup):
    def __init__(
        self,
        uv_surface: OpenGLSurface,
        resolution: tuple[int, int] | None = None,
        stroke_width: float = 1,
        normal_nudge: float = 1e-2,
        depth_test: bool = True,
        flat_stroke: bool = False,
        **kwargs: Any,
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
            **kwargs,
        )

    def init_points(self) -> None:
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
