from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
from skimage import measure

from manim import *
from manim.mobject.three_d.three_dimensions import ThreeDVMobject
from manim.utils.color import ManimColor


class ImplicitSurface(ThreeDVMobject):
    """Render an implicit isosurface using the Marching Cubes algorithm.

    Parameters
    ----------
    func : Callable[[float | np.ndarray, float | np.ndarray, float | np.ndarray], float | np.ndarray]
        Implicit function f(x, y, z). The surface is defined where f(x, y, z) = isolevel.
    resolution : int, default=25
        Number of divisions per axis.
    isolevel : float, default=0.0
        The isosurface level.
    x_range, y_range, z_range : Sequence[float], default=(-2, 2)
        Sampling ranges.
    color : Color, default=BLUE
        Color of the surface.
    **kwargs
        Additional arguments passed to ThreeDVMobject.
    """

    def __init__(
        self,
        func: Callable[
            [float | np.ndarray, float | np.ndarray, float | np.ndarray],
            float | np.ndarray,
        ],
        resolution: int = 25,
        isolevel: float = 0.0,
        x_range: Sequence[float] = (-2.0, 2.0),
        y_range: Sequence[float] = (-2.0, 2.0),
        z_range: Sequence[float] = (-2.0, 2.0),
        color: ManimColor = BLUE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Generates the 3D grid:
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        z = np.linspace(z_range[0], z_range[1], resolution)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        values = func(X, Y, Z)

        # Extracts the mesh:
        verts, faces, _, _ = measure.marching_cubes(values, level=isolevel)

        # Normalizes to the real domain:
        scale_x = (x_range[1] - x_range[0]) / resolution
        scale_y = (y_range[1] - y_range[0]) / resolution
        scale_z = (z_range[1] - z_range[0]) / resolution
        verts = np.array(
            [
                [
                    x_range[0] + v[0] * scale_x,
                    y_range[0] + v[1] * scale_y,
                    z_range[0] + v[2] * scale_z,
                ]
                for v in verts
            ]
        )

        # Builds the polygons
        for face in faces:
            tri = [verts[i] for i in face]
            self.add(Polygon(*tri, color=color, fill_opacity=1))
