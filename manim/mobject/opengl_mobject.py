from manim.mobject.mobject import Mobject
import numpy as np
from ..constants import ORIGIN


class OpenGLPoint(Mobject):
    def __init__(
        self, location=ORIGIN, artificial_width=1e-6, artificial_height=1e-6, **kwargs
    ):
        self.artificial_width = artificial_width
        self.artificial_height = artificial_height
        super().__init__(self, **kwargs)
        self.set_location(location)

    def get_width(self):
        return self.artificial_width

    def get_height(self):
        return self.artificial_height

    def get_location(self):
        return self.points[0].copy()

    def get_bounding_box_point(self, *args, **kwargs):
        return self.get_location()

    def set_location(self, new_loc):
        self.set_points(np.array(new_loc, ndmin=2, dtype=float))
