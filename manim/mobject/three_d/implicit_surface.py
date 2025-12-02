"""Three-dimensional implicit surfaces."""

from __future__ import annotations

__all__ = ["ImplicitSurface"]

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import mcubes
import numpy as np

from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.three_d.three_dimensions import ThreeDVMobject
from manim.mobject.types.vectorized_mobject import VGroup
from manim.utils.color import BLUE, ManimColor, ParsableManimColor

if TYPE_CHECKING:
    pass


class ImplicitSurface(VGroup, metaclass=ConvertToOpenGL):
    """A 3D implicit surface defined by f(x, y, z) = 0.

    This class creates a triangular mesh representation of an implicit surface
    using the marching cubes algorithm. The surface is defined by a scalar
    function f(x, y, z), and the class extracts the isosurface where f = level.

    Parameters
    ----------
    func
        A callable that takes three arguments (x, y, z) and returns a scalar value.
        The function should be NumPy-aware for efficient evaluation over grids.
        The surface is extracted where func(x, y, z) = level.
    x_range
        The range of x values as (x_min, x_max). Defaults to (-1.0, 1.0).
    y_range
        The range of y values as (y_min, y_max). Defaults to (-1.0, 1.0).
    z_range
        The range of z values as (z_min, z_max). Defaults to (-1.0, 1.0).
    resolution
        The number of sample points along each axis. Can be an integer
        (same resolution for all axes) or a 3-tuple (nx, ny, nz).
        Higher values give smoother surfaces but take longer to compute.
        Defaults to 32.
    level
        The isosurface level to extract. The surface is where
        func(x, y, z) = level. Defaults to 0.0.
    fill_color
        The fill color of the surface faces. Defaults to BLUE.
    fill_opacity
        The opacity of the surface faces, from 0 (transparent) to 1 (opaque).
        Defaults to 1.0.
    stroke_color
        The stroke color of the triangle edges. If None, no stroke is applied.
    stroke_width
        The width of the triangle edge strokes. Defaults to 0.0 (no stroke).
    faces_config
        Additional keyword arguments passed to each triangle face.

    Examples
    --------
    .. manim:: ImplicitSphereExample
        :save_last_frame:

        class ImplicitSphereExample(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes(
                    x_range=(-1.5, 1.5, 1),
                    y_range=(-1.5, 1.5, 1),
                    z_range=(-1.5, 1.5, 1),
                )

                # Unit sphere: x^2 + y^2 + z^2 = 1
                def sphere_func(x, y, z):
                    return x**2 + y**2 + z**2 - 1.0

                surface = ImplicitSurface(
                    sphere_func,
                    x_range=(-1.3, 1.3),
                    y_range=(-1.3, 1.3),
                    z_range=(-1.3, 1.3),
                    resolution=40,
                    fill_color=BLUE,
                    fill_opacity=0.8,
                )

                self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
                self.add(axes, surface)

    .. manim:: ImplicitTorusExample
        :save_last_frame:

        class ImplicitTorusExample(ThreeDScene):
            def construct(self):
                # Torus with major radius 2, minor radius 0.5
                R, r = 2.0, 0.5

                def torus_func(x, y, z):
                    return (np.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2

                surface = ImplicitSurface(
                    torus_func,
                    x_range=(-3, 3),
                    y_range=(-3, 3),
                    z_range=(-1, 1),
                    resolution=50,
                    fill_color=GREEN,
                    fill_opacity=0.9,
                )

                self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
                self.add(surface)

    Notes
    -----
    The function should be NumPy-aware and support broadcasting. If your function
    only accepts scalar inputs, you can wrap it with ``np.vectorize``::

        def scalar_f(x, y, z):
            # some computation that only works with scalars
            return result


        # Wrap for use with ImplicitSurface
        surface = ImplicitSurface(np.vectorize(scalar_f), ...)

    However, vectorized functions are slower than native NumPy implementations.

    See Also
    --------
    :class:`.ImplicitFunction` : 2D implicit curves defined by f(x, y) = 0.
    :class:`.Surface` : Parametric surfaces defined by (u, v) -> (x, y, z).
    """

    def __init__(
        self,
        func: Callable[[float, float, float], float],
        x_range: Sequence[float] = (-1.0, 1.0),
        y_range: Sequence[float] = (-1.0, 1.0),
        z_range: Sequence[float] = (-1.0, 1.0),
        resolution: int | Sequence[int] = 32,
        level: float = 0.0,
        fill_color: ParsableManimColor = BLUE,
        fill_opacity: float = 1.0,
        stroke_color: ParsableManimColor | None = None,
        stroke_width: float = 0.0,
        faces_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.func = func
        self.x_range = tuple(x_range)
        self.y_range = tuple(y_range)
        self.z_range = tuple(z_range)
        self.level = level
        self.resolution = self._normalize_resolution(resolution)

        # Build face style configuration
        self._fill_color = ManimColor(fill_color)
        self._fill_opacity = fill_opacity
        self._stroke_color = ManimColor(stroke_color) if stroke_color else None
        self._stroke_width = stroke_width
        self._faces_config = faces_config or {}

        self._build_surface()

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

    def _build_surface(self) -> None:
        """Build the triangular mesh from the implicit function."""
        nx, ny, nz = self.resolution

        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range

        # Create grid coordinates
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        zs = np.linspace(z_min, z_max, nz)

        # Create meshgrid with indexing="ij" so axis 0 -> x, axis 1 -> y, axis 2 -> z
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

        # Evaluate the function on the grid
        values = np.asarray(self.func(X, Y, Z), dtype=np.float64)
        if values.shape != (nx, ny, nz):
            raise ValueError(
                f"func must return array of shape {(nx, ny, nz)}, "
                f"got {values.shape} instead. Ensure your function is NumPy-aware "
                "and supports broadcasting."
            )

        # Check if there's actually a surface to extract
        if np.all(values > self.level) or np.all(values < self.level):
            # No surface crosses through the domain
            return

        # Run marching cubes algorithm using PyMCubes
        try:
            vertices, triangles = mcubes.marching_cubes(values, self.level)
        except Exception:
            # marching_cubes can fail if no surface is found
            return

        if len(vertices) == 0 or len(triangles) == 0:
            return

        # Map grid indices in vertices to world coordinates
        # vertices[:, 0] is in [0, nx-1], etc.
        vertices[:, 0] = x_min + vertices[:, 0] * (x_max - x_min) / (nx - 1)
        vertices[:, 1] = y_min + vertices[:, 1] * (y_max - y_min) / (ny - 1)
        vertices[:, 2] = z_min + vertices[:, 2] * (z_max - z_min) / (nz - 1)

        # Store mesh data
        self._mesh_vertices = vertices
        self._mesh_triangles = triangles

        # Build triangle faces using ThreeDVMobject
        for tri in triangles:
            p1, p2, p3 = vertices[tri]
            # Create a closed triangle path: p1 -> p2 -> p3 -> p1
            face = ThreeDVMobject()
            face.set_points_as_corners([p1, p2, p3, p1])
            face.set_fill(
                color=self._fill_color,
                opacity=self._fill_opacity,
            )
            face.set_stroke(
                color=self._stroke_color,
                width=self._stroke_width,
            )
            self.add(face)

    def get_vertices(self) -> np.ndarray:
        """Return all unique vertices of the surface mesh.

        Returns
        -------
        np.ndarray
            An array of shape (N, 3) containing the 3D coordinates
            of all unique vertices in the mesh.
        """
        if hasattr(self, "_mesh_vertices") and self._mesh_vertices is not None:
            return self._mesh_vertices.copy()

        return np.array([]).reshape(0, 3)

    def verify_surface(
        self,
        n_samples: int = 500,
    ) -> dict[str, float]:
        """Verify that vertices lie on the implicit surface.

        This method samples vertices from the surface and checks how close
        they are to satisfying f(x, y, z) = level.

        Parameters
        ----------
        n_samples
            Number of vertices to sample for verification. Defaults to 500.

        Returns
        -------
        dict
            A dictionary with 'mean_error' and 'max_error' keys containing
            the mean and maximum absolute deviation from the level value.

        Examples
        --------
        >>> def sphere(x, y, z):
        ...     return x**2 + y**2 + z**2 - 1
        >>> surface = ImplicitSurface(sphere, resolution=20)
        >>> errors = surface.verify_surface()
        >>> errors['mean_error'] < 0.01  # Verify accuracy
        True
        >>> errors['max_error'] < 0.02
        True
        """
        vertices = self.get_vertices()
        if len(vertices) == 0:
            return {"mean_error": 0.0, "max_error": 0.0}

        # Sample vertices if there are many
        if len(vertices) > n_samples:
            indices = np.random.choice(len(vertices), size=n_samples, replace=False)
            vertices = vertices[indices]

        # Evaluate function at vertices
        values = self.func(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        errors = np.abs(values - self.level)

        return {
            "mean_error": float(np.mean(errors)),
            "max_error": float(np.max(errors)),
        }
