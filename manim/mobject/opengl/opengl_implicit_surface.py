"""OpenGL implementation of implicit surfaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import mcubes
import moderngl
import numpy as np

from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.utils.color import GREY

if TYPE_CHECKING:
    pass

__all__ = ["OpenGLImplicitSurface"]


class OpenGLImplicitSurface(OpenGLMobject):
    """OpenGL implementation of a 3D implicit surface.

    This class creates a triangular mesh representation of an implicit surface
    using the marching cubes algorithm, optimized for OpenGL rendering.

    Parameters
    ----------
    func
        A callable that takes three arguments (x, y, z) and returns a scalar value.
        The function should be NumPy-aware for efficient evaluation over grids.
    x_range
        The range of x values as (x_min, x_max). Defaults to (-1.0, 1.0).
    y_range
        The range of y values as (y_min, y_max). Defaults to (-1.0, 1.0).
    z_range
        The range of z values as (z_min, z_max). Defaults to (-1.0, 1.0).
    resolution
        The number of sample points along each axis. Can be an integer
        (same resolution for all axes) or a 3-tuple (nx, ny, nz).
        Defaults to 32.
    level
        The isosurface level to extract. Defaults to 0.0.
    color
        The color of the surface. Defaults to GREY.
    opacity
        The opacity of the surface. Defaults to 1.0.
    gloss
        The glossiness of the surface. Defaults to 0.3.
    shadow
        The shadow intensity. Defaults to 0.4.
    """

    shader_dtype = [
        ("point", np.float32, (3,)),
        ("normal", np.float32, (3,)),
        ("color", np.float32, (4,)),
    ]
    shader_folder = "surface"

    def __init__(
        self,
        func: Callable[[float, float, float], float] | None = None,
        x_range: Sequence[float] = (-1.0, 1.0),
        y_range: Sequence[float] = (-1.0, 1.0),
        z_range: Sequence[float] = (-1.0, 1.0),
        resolution: int | Sequence[int] = 32,
        level: float = 0.0,
        color=GREY,
        opacity: float = 1.0,
        gloss: float = 0.3,
        shadow: float = 0.4,
        render_primitive=moderngl.TRIANGLES,
        depth_test: bool = True,
        **kwargs,
    ):
        self.func = func
        self.x_range = tuple(x_range)
        self.y_range = tuple(y_range)
        self.z_range = tuple(z_range)
        self.level = level
        self.resolution = self._normalize_resolution(resolution)

        # Store mesh data
        self._mesh_vertices: np.ndarray | None = None
        self._mesh_triangles: np.ndarray | None = None
        self._triangle_indices: np.ndarray | None = None

        super().__init__(
            color=color,
            opacity=opacity,
            gloss=gloss,
            shadow=shadow,
            render_primitive=render_primitive,
            depth_test=depth_test,
            **kwargs,
        )

    @staticmethod
    def _normalize_resolution(resolution: int | Sequence[int]) -> tuple[int, int, int]:
        """Convert resolution to a 3-tuple of integers."""
        if isinstance(resolution, int):
            if resolution < 2:
                raise ValueError("resolution must be >= 2")
            return (resolution, resolution, resolution)

        res = tuple(resolution)
        if len(res) == 1:
            if res[0] < 2:
                raise ValueError("resolution must be >= 2")
            return (res[0], res[0], res[0])
        if len(res) != 3:
            raise ValueError("resolution must be an int or a 3-tuple")
        if any(r < 2 for r in res):
            raise ValueError("each resolution component must be >= 2")
        return res  # type: ignore[return-value]

    def init_points(self):
        """Initialize the surface mesh using marching cubes."""
        if self.func is None:
            self.set_points(np.zeros((0, 3)))
            self._triangle_indices = np.zeros(0, dtype=int)
            return

        nx, ny, nz = self.resolution
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range

        # Create grid
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        zs = np.linspace(z_min, z_max, nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        # Evaluate function
        values = np.asarray(self.func(X, Y, Z), dtype=np.float64)

        # Check if there's a surface
        if np.all(values > self.level) or np.all(values < self.level):
            self.set_points(np.zeros((0, 3)))
            self._triangle_indices = np.zeros(0, dtype=int)
            return

        try:
            vertices, triangles = mcubes.marching_cubes(values, self.level)
        except Exception:
            self.set_points(np.zeros((0, 3)))
            self._triangle_indices = np.zeros(0, dtype=int)
            return

        if len(vertices) == 0:
            self.set_points(np.zeros((0, 3)))
            self._triangle_indices = np.zeros(0, dtype=int)
            return

        # Map to world coordinates
        vertices[:, 0] = x_min + vertices[:, 0] * (x_max - x_min) / (nx - 1)
        vertices[:, 1] = y_min + vertices[:, 1] * (y_max - y_min) / (ny - 1)
        vertices[:, 2] = z_min + vertices[:, 2] * (z_max - z_min) / (nz - 1)

        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        # Flatten triangles for rendering
        all_verts = vertices[triangles.flatten()]
        self._triangle_indices = np.arange(len(all_verts), dtype=int)

        self.set_points(all_verts)

    def get_triangle_indices(self):
        """Return indices for triangle rendering."""
        if self._triangle_indices is None:
            return np.zeros(0, dtype=int)
        return self._triangle_indices

    def get_shader_vert_indices(self):
        """Return vertex indices for shader."""
        return self.get_triangle_indices()

    def _compute_normals(self) -> np.ndarray:
        """Compute normals for each vertex based on face normals."""
        points = self.points
        if len(points) == 0:
            return np.zeros((0, 3))

        normals = np.zeros_like(points)

        # Process triangles (every 3 points is a triangle)
        for i in range(0, len(points), 3):
            if i + 2 >= len(points):
                break
            p0, p1, p2 = points[i], points[i + 1], points[i + 2]
            # Compute face normal
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            # Assign same normal to all 3 vertices
            normals[i] = normal
            normals[i + 1] = normal
            normals[i + 2] = normal

        return normals

    def get_shader_data(self):
        """Generate shader data for rendering."""
        points = self.points
        if len(points) == 0:
            return np.zeros(0, dtype=self.shader_dtype)

        shader_data = np.zeros(len(points), dtype=self.shader_dtype)
        shader_data["point"] = points

        # Compute and add normals
        normals = self._compute_normals()
        shader_data["normal"] = normals

        # Add colors
        self.fill_in_shader_color_info(shader_data)

        return shader_data

    def fill_in_shader_color_info(self, shader_data):
        """Fill color information in shader data."""
        self.read_data_to_shader(shader_data, "color", "rgbas")
        return shader_data
