__all__ = ["TrueDot", "DotCloud"]

import numpy as np

from ...constants import ORIGIN, RIGHT, UP
from ...utils.color import YELLOW
from .opengl_point_cloud_mobject import OpenGLPMObject

DEFAULT_GRID_HEIGHT = 6
DEFAULT_BUFF_RATIO = 0.5
DEFAULT_POINT_DENSITY_1D: int = 10


class DotCloud(OpenGLPMObject):
    def __init__(
        self,
        points=None,
        color=YELLOW,
        stroke_width=2.0,
        radius=2.0,
        density=DEFAULT_POINT_DENSITY_1D,
        **kwargs
    ):
        self.radius = radius
        self.epsilon = 1.0 / density
        super().__init__(stroke_width=stroke_width, color=color, **kwargs)
        if points is not None:
            self.set_points(points)

    # def init_data(self):
    #     super().init_data()
    #     self.data["radii"] = np.zeros((1, 1))
    #     self.set_radius(self.stroke_width)

    def init_points(self):
        self.reset_points()
        self.data["points"] = np.array(self.generate_points(), dtype=np.float32)

    def to_grid(
        self,
        n_rows,
        n_cols,
        n_layers=1,
        buff_ratio=None,
        h_buff_ratio=1.0,
        v_buff_ratio=1.0,
        d_buff_ratio=1.0,
        height=DEFAULT_GRID_HEIGHT,
    ):
        n_points = n_rows * n_cols * n_layers
        points = np.repeat(range(n_points), 3, axis=0).reshape((n_points, 3))
        points[:, 0] = points[:, 0] % n_cols
        points[:, 1] = (points[:, 1] // n_cols) % n_rows
        points[:, 2] = points[:, 2] // (n_rows * n_cols)
        self.set_points(points.astype(float))

        if buff_ratio is not None:
            v_buff_ratio = buff_ratio
            h_buff_ratio = buff_ratio
            d_buff_ratio = buff_ratio

        radius = self.get_radius()
        ns = [n_cols, n_rows, n_layers]
        brs = [h_buff_ratio, v_buff_ratio, d_buff_ratio]
        self.set_radius(0)
        for n, br, dim in zip(ns, brs, range(3)):
            self.rescale_to_fit(2 * radius * (1 + br) * (n - 1), dim, stretch=True)
        self.set_radius(radius)
        if height is not None:
            self.set_height(height)
        self.center()
        return self

    def generate_points(self):
        return [
            r * (np.cos(theta) * RIGHT + np.sin(theta) * UP)
            for r in np.arange(self.epsilon, self.radius, self.epsilon)
            # Num is equal to int(stop - start)/ (step + 1) reformulated.
            for theta in np.linspace(
                0, 2 * np.pi, num=int(2 * np.pi * (r + self.epsilon) / self.epsilon)
            )
        ]

    # def set_radii(self, radii):
    #     self.data["radii"][:] = resize_preserving_order(radii, len(self.data["radii"]))
    #     self.refresh_bounding_box()
    #     return self

    # def get_radii(self):
    #     return self.data["radii"]

    # def set_radius(self, radius):
    #     self.data["radii"][:] = radius
    #     self.refresh_bounding_box()
    #     return self

    # def get_radius(self):
    #     return self.get_radii().max()

    # def compute_bounding_box(self):
    #     bb = super().compute_bounding_box()
    #     radius = self.get_radius()
    #     bb[0] += np.full((3,), -radius)
    #     bb[2] += np.full((3,), radius)
    #     return bb

    # def scale(self, scale_factor, scale_radii=True, **kwargs):
    #     super().scale(scale_factor, **kwargs)
    #     if scale_radii:
    #         self.set_radii(scale_factor * self.get_radii())
    #     return self

    def make_3d(self, gloss=0.5, shadow=0.2):
        self.set_gloss(gloss)
        self.set_shadow(shadow)
        self.apply_depth_test()
        return self


class TrueDot(DotCloud):
    def __init__(self, center=ORIGIN, stroke_width=2.0, **kwargs):
        self.radius = stroke_width
        super().__init__(points=[center], stroke_width=stroke_width, **kwargs)
