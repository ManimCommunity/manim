"""Mobjects representing point clouds."""

__all__ = ["PMobject", "Mobject1D", "Mobject2D", "PGroup", "PointCloudDot", "Point"]


import numpy as np

from ...constants import *
from ...mobject.mobject import Mobject
from ...utils.bezier import interpolate
from ...utils.color import (
    BLACK,
    WHITE,
    YELLOW,
    YELLOW_C,
    color_gradient,
    color_to_rgba,
    rgba_to_color,
)
from ...utils.iterables import stretch_array_to_length


class PMobject(Mobject):
    def __init__(self, stroke_width=DEFAULT_STROKE_WIDTH, **kwargs):
        self.stroke_width = stroke_width
        super().__init__(**kwargs)

    def reset_points(self):
        self.rgbas = np.zeros((0, 4))
        self.points = np.zeros((0, 3))
        return self

    def get_array_attrs(self):
        return Mobject.get_array_attrs(self) + ["rgbas"]

    def add_points(self, points, rgbas=None, color=None, alpha=1):
        """
        Points must be a Nx3 numpy array, as must rgbas if it is not None
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        num_new_points = len(points)
        self.points = np.append(self.points, points, axis=0)
        if rgbas is None:
            color = Color(color) if color else self.color
            rgbas = np.repeat([color_to_rgba(color, alpha)], num_new_points, axis=0)
        elif len(rgbas) != len(points):
            raise ValueError("points and rgbas must have same shape")
        self.rgbas = np.append(self.rgbas, rgbas, axis=0)
        return self

    def set_color(self, color=YELLOW_C, family=True):
        rgba = color_to_rgba(color)
        mobs = self.family_members_with_points() if family else [self]
        for mob in mobs:
            mob.rgbas[:, :] = rgba
        self.color = color
        return self

    def get_stroke_width(self):
        return self.stroke_width

    def set_stroke_width(self, width, family=True):
        mobs = self.family_members_with_points() if family else [self]
        for mob in mobs:
            mob.stroke_width = width
        return self

    # def set_color_by_gradient(self, start_color, end_color):
    def set_color_by_gradient(self, *colors):
        self.rgbas = np.array(
            list(map(color_to_rgba, color_gradient(colors, len(self.points))))
        )
        return self

    def set_colors_by_radial_gradient(
        self, center=None, radius=1, inner_color=WHITE, outer_color=BLACK
    ):
        start_rgba, end_rgba = list(map(color_to_rgba, [start_color, end_color]))
        if center is None:
            center = self.get_center()
        for mob in self.family_members_with_points():
            num_points = mob.get_num_points()
            t = min(1, np.abs(mob.get_center() - center) / radius)

            mob.rgbas = np.array([interpolate(start_rgba, end_rgba, t)] * num_points)
        return self

    def match_colors(self, mobject):
        Mobject.align_data(self, mobject)
        self.rgbas = np.array(mobject.rgbas)
        return self

    def filter_out(self, condition):
        for mob in self.family_members_with_points():
            to_eliminate = ~np.apply_along_axis(condition, 1, mob.points)
            mob.points = mob.points[to_eliminate]
            mob.rgbas = mob.rgbas[to_eliminate]
        return self

    def thin_out(self, factor=5):
        """
        Removes all but every nth point for n = factor
        """
        for mob in self.family_members_with_points():
            num_points = self.get_num_points()
            mob.apply_over_attr_arrays(
                lambda arr: arr[np.arange(0, num_points, factor)]
            )
        return self

    def sort_points(self, function=lambda p: p[0]):
        """
        Function is any map from R^3 to R
        """
        for mob in self.family_members_with_points():
            indices = np.argsort(np.apply_along_axis(function, 1, mob.points))
            mob.apply_over_attr_arrays(lambda arr: arr[indices])
        return self

    def fade_to(self, color, alpha):
        self.rgbas = interpolate(self.rgbas, color_to_rgba(color), alpha)
        for mob in self.submobjects:
            mob.fade_to(color, alpha)
        return self

    def get_all_rgbas(self):
        return self.get_merged_array("rgbas")

    def ingest_submobjects(self):
        attrs = self.get_array_attrs()
        arrays = list(map(self.get_merged_array, attrs))
        for attr, array in zip(attrs, arrays):
            setattr(self, attr, array)
        self.submobjects = []
        return self

    def get_color(self):
        return rgba_to_color(self.rgbas[0, :])

    def point_from_proportion(self, alpha):
        index = alpha * (self.get_num_points() - 1)
        return self.points[index]

    # Alignment
    def align_points_with_larger(self, larger_mobject):
        assert isinstance(larger_mobject, PMobject)
        self.apply_over_attr_arrays(
            lambda a: stretch_array_to_length(a, larger_mobject.get_num_points())
        )

    def get_point_mobject(self, center=None):
        if center is None:
            center = self.get_center()
        return Point(center)

    def interpolate_color(self, mobject1, mobject2, alpha):
        self.rgbas = interpolate(mobject1.rgbas, mobject2.rgbas, alpha)
        self.set_stroke_width(
            interpolate(
                mobject1.get_stroke_width(),
                mobject2.get_stroke_width(),
                alpha,
            )
        )
        return self

    def pointwise_become_partial(self, mobject, a, b):
        lower_index, upper_index = [int(x * mobject.get_num_points()) for x in (a, b)]
        for attr in self.get_array_attrs():
            full_array = getattr(mobject, attr)
            partial_array = full_array[lower_index:upper_index]
            setattr(self, attr, partial_array)


# TODO, Make the two implementations below non-redundant
class Mobject1D(PMobject):
    def __init__(self, density=DEFAULT_POINT_DENSITY_1D, **kwargs):
        self.density = density
        self.epsilon = 1.0 / self.density
        PMobject.__init__(self, **kwargs)

    def add_line(self, start, end, color=None):
        start, end = list(map(np.array, [start, end]))
        length = np.linalg.norm(end - start)
        if length == 0:
            points = [start]
        else:
            epsilon = self.epsilon / length
            points = [interpolate(start, end, t) for t in np.arange(0, 1, epsilon)]
        self.add_points(points, color=color)


class Mobject2D(PMobject):
    def __init__(self, density=DEFAULT_POINT_DENSITY_2D, **kwargs):
        self.density = density
        self.epsilon = 1.0 / self.density
        PMobject.__init__(self, **kwargs)


class PGroup(PMobject):
    def __init__(self, *pmobs, **kwargs):
        if not all([isinstance(m, PMobject) for m in pmobs]):
            raise ValueError("All submobjects must be of type PMobject")
        super().__init__(**kwargs)
        self.add(*pmobs)


class PointCloudDot(Mobject1D):
    """A disc made of a cloud of Dots
    Example
    --------
    .. manim:: PointCloudDotExample
        :save_last_frame:

        class PointCloudDotExample(Scene):
            def construct(self):
                cloud_1 = PointCloudDot(color=RED)
                cloud_2 = PointCloudDot(stroke_width=4, radius=1)
                cloud_3 = PointCloudDot(density=15)

                group = Group(cloud_1, cloud_2, cloud_3).arrange()
                self.add(group)

    .. manim:: PointCloudDotExample2

        class PointCloudDotExample2(Scene):
            def construct(self):
                plane = ComplexPlane()
                cloud = PointCloudDot(color=RED)
                self.add(
                    plane, cloud
                )
                self.wait()
                self.play(
                    cloud.animate.apply_complex_function(lambda z: np.exp(z))
                )
    """

    def __init__(
        self,
        center=ORIGIN,
        radius=2.0,
        stroke_width=2,
        density=DEFAULT_POINT_DENSITY_1D,
        color=YELLOW,
        **kwargs
    ):
        self.radius = radius
        Mobject1D.__init__(
            self,
            radius=radius,
            stroke_width=stroke_width,
            density=density,
            color=color,
            **kwargs
        )
        self.shift(center)

    def generate_points(self):
        self.add_points(
            [
                r * (np.cos(theta) * RIGHT + np.sin(theta) * UP)
                for r in np.arange(self.epsilon, self.radius, self.epsilon)
                # Num is equal to int(stop - start)/ (step + 1) reformulated.
                for theta in np.linspace(
                    0, 2 * np.pi, num=int(2 * np.pi * (r + self.epsilon) / self.epsilon)
                )
            ]
        )


class Point(PMobject):
    def __init__(self, location=ORIGIN, color=BLACK, **kwargs):
        PMobject.__init__(self, color=color, **kwargs)
        self.add_points([location])
