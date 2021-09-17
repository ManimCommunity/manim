from typing import Callable, Sequence

import numpy as np

from .. import config
from .contour import _contour
from .opengl_compatibility import ConvertToOpenGL
from .types.vectorized_mobject import VMobject

__all__ = ["ImplicitFunction"]


class _Contour:
    def __init__(self, z):
        z = np.ma.asarray(z, dtype=np.float64)
        origin = (0, 0)
        step = (1, 1)
        y, x = np.mgrid[
            origin[0] : (origin[0] + step[0] * z.shape[0]) : step[0],
            origin[1] : (origin[1] + step[1] * z.shape[1]) : step[1],
        ]
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.ma.asarray(z, dtype=np.float64)
        self._contour_generator = _contour.QuadContourGenerator(
            x,
            y,
            z.filled(),
            None,
            True,
            0,
        )

    def contour(self, level):
        return self._contour_generator.create_contour(level)


class ImplicitFunction(VMobject, metaclass=ConvertToOpenGL):
    def __init__(
        self,
        func: Callable[[float, float], float],
        x_range: Sequence[float] = None,
        y_range: Sequence[float] = None,
        res: int = 100,
        **kwargs
    ):
        """An implicit function.

        Parameters
        ----------
        func
            The implicit function in the form of ``f(x, y) = 0``
        x_range
            The x min and max boundary for the graph
        y_range
            The y min and max boundary for the graph
        res
            The resolution of the implicit graph
        kwargs
            Additional parameters to be passed to :class:`VMobject`
        """
        super().__init__(**kwargs)
        x_range = x_range or [
            -config.frame_width / 2,
            config.frame_width / 2,
        ]
        y_range = y_range or [
            -config.frame_height / 2,
            config.frame_height / 2,
        ]
        x, y = (
            np.arange(x_range[0], x_range[1] + 1 / res, 1 / res),
            np.arange(y_range[0], y_range[1] + 1 / res, 1 / res),
        )
        z = func(x[np.newaxis, :], y[:, np.newaxis])
        c = _Contour(z)
        a = [[np.append(j, [0]) for j in i] for i in c.contour(0)]
        self.set_points_as_corners(a[0])
        for b in a[1:]:
            self.add(self.copy().set_points_as_corners(b))
        x0, y0 = (x_range[1] - x_range[0]) * 50, (y_range[1] - y_range[0]) * 50
        self.scale(1 / res, about_point=[x0, y0, 0])
        self.shift([-x0, -y0, 0])
