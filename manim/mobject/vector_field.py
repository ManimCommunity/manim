"""Mobjects representing vector fields."""

__all__ = [
    "VectorField",
    "ArrowVectorField",
    "StreamLines",
    "ShowPassingFlashWithThinningStrokeWidth",
    "AnimatedStreamLines",
    "get_colored_background_image",
    "get_color_gradient_function",
    "get_color_field_image_file",
]

import itertools as it
import os
import random
from math import ceil, floor
from typing import Callable, Optional, Sequence

import numpy as np
from colour import Color
from PIL import Image

from .. import config, logger
from ..animation.composition import AnimationGroup
from ..animation.indication import ShowPassingFlash
from ..constants import *
from ..mobject.geometry import Vector
from ..mobject.mobject import Mobject
from ..mobject.types.vectorized_mobject import VGroup, VMobject
from ..utils.bezier import interpolate, inverse_interpolate
from ..utils.color import (
    BLUE,
    BLUE_E,
    GREEN,
    RED,
    WHITE,
    YELLOW,
    Colors,
    color_to_rgb,
    rgb_to_color,
)
from ..utils.rate_functions import linear
from ..utils.simple_functions import sigmoid
from ..utils.space_ops import get_norm

# from ..utils.space_ops import normalize


DEFAULT_SCALAR_FIELD_COLORS: list = [BLUE_E, GREEN, YELLOW, RED]


def get_colored_background_image(
    scalar_field_func: Callable,  # TODO: What is taken as parameters?
    number_to_rgb_func: Callable,
) -> Image:
    ph = config["pixel_height"]
    pw = config["pixel_width"]
    fw = config["frame_width"]
    fh = config["frame_height"]
    points_array = np.zeros((ph, pw, 3))
    x_array = np.linspace(-fw / 2, fw / 2, pw)
    x_array = x_array.reshape((1, len(x_array)))
    x_array = x_array.repeat(ph, axis=0)

    y_array = np.linspace(fh / 2, -fh / 2, ph)
    y_array = y_array.reshape((len(y_array), 1))
    y_array.repeat(pw, axis=1)
    points_array[:, :, 0] = x_array
    points_array[:, :, 1] = y_array
    scalars = np.apply_along_axis(scalar_field_func, 2, points_array)
    rgb_array = number_to_rgb_func(scalars.flatten()).reshape((ph, pw, 3))
    return Image.fromarray((rgb_array * 255).astype("uint8"))


def get_color_gradient_function(
    min_value: float = 0,
    max_value: float = 1,
    colors: list = [BLUE, RED],
) -> Callable[[float], Color]:
    rgbs = np.array(list(map(color_to_rgb, colors)))

    def get_interpolated_color(value: float):
        alpha = (value - min_value) / float(max_value - min_value)
        alpha = np.clip(alpha, 0, 1) * (len(rgbs) - 1)
        color1 = rgbs[int(alpha)]
        color2 = rgbs[min(int(alpha + 1), len(rgbs) - 1)]
        alpha %= 1
        rgb = (1 - alpha) * color1 + alpha * color2
        return rgb_to_color(rgb)

    return get_interpolated_color


# TODO: RASTER_IMAGE_DIR is undefined. Therefor this function doesn't work
def get_color_field_image_file(
    scalar_func: Callable[[np.ndarray], np.ndarray],
    min_value: int = 0,
    max_value: int = 2,
    colors: list = DEFAULT_SCALAR_FIELD_COLORS,
) -> str:
    # try_hash
    np.random.seed(0)
    sample_inputs = 5 * np.random.random(size=(10, 3)) - 10
    sample_outputs = np.apply_along_axis(scalar_func, 1, sample_inputs)
    func_hash = hash(
        str(min_value) + str(max_value) + str(colors) + str(sample_outputs)
    )
    file_name = "%d.png" % func_hash
    full_path = os.path.join(RASTER_IMAGE_DIR, file_name)
    if not os.path.exists(full_path):
        logger.info("Rendering color field image " + str(func_hash))
        rgb_gradient_func = get_color_gradient_function(
            min_value=min_value, max_value=max_value, colors=colors
        )
        image = get_colored_background_image(scalar_func, rgb_gradient_func)
        image.save(full_path)
    return full_path


# Mobjects


class VectorField(VGroup):
    def __init__(self, func: Callable[[np.ndarray], np.ndarray], **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.submob_movement_updater = None

    def nudge(self, mob, dt=1, substeps=1, pointwise=False):
        step_size = dt / substeps
        for i in range(substeps):
            if pointwise:
                mob.apply_function(lambda p: p + self.func(p) * step_size)
            else:
                mob.shift(self.func(mob.get_center()) * step_size)

    def nudge_submobjects(self, dt=1, substeps=1, pointwise=False):
        for mob in self.submobjects:
            self.nudge(mob, dt, substeps, pointwise)

    def get_nudge_updater(self, speed=1, pointwise=False):
        return lambda mob, dt: self.nudge(mob, dt * speed)

    def start_submobject_movement(self, speed=1, pointwise=False):
        self.stop_submobject_movement()
        self.submob_movement_updater = lambda mob, dt: mob.nudge_submobjects(
            dt * speed, pointwise=pointwise
        )
        self.add_updater(self.submob_movement_updater)

    def stop_submobject_movement(self):
        self.remove_updater(self.submob_movement_updater)
        self.submob_movement_updater = None

    # def scale(self, scale_factor: float, **kwargs) -> "Mobject":
    #     return super().scale(scale_factor, **kwargs)

    # def shift(self, *vectors: np.ndarray) -> "Mobject":
    #     return super().shift(*vectors)


class ArrowVectorField(VectorField):
    """A :class:`VectorField` represented by a set of change vectors.

    `VectorField`s are allways based on a function defining the vector at every position.
    This the values of this functions is displayed as a grid of vectors.
    The color of each vector is determined by it's magnitude.
    A color gradient can be used to color the vectors in a defined interval of magnitudes.

    Parameters
    ----------
    func
        The function defining the rate of change at every position of the `VectorField`.
    delta_x
        The distance in x direction between two vectors.
    delta_y
        The distance in y direction between two vectors.
    min_magnitude
        The magnitude at which the color gradient starts. Every vector with lower magnitude is colored with the first color in the gradient.
    max_magnitude
        The magnitude at which the color gradient ends. Every vector with bigger magnitude is colored with the last color in the gradient.
    colors
        The colors used as color gradient.
    length_func
        The function determining the displayed size of the vectors. The actual size
        of the vector is passed, the returned value will be used as display size for the
        vector. By default this is used to cap the displayed size of vectors to reduce the clutter.
    opacity
        The opacity of the arrows.
    vector_config
        Additional arguments to be passed to the :class:`~.Vector`-constructor
    kwargs : Any
        Additional arguments to be passed to the :class:`~.VGroup`-constructor

    Examples
    --------

    .. manim:: BasicUsage
        :save_last_frame:

        class BasicUsage(Scene):
            def construct(self):
                func = lambda pos: pos[1]*RIGHT/2+pos[0]*UP/3
                self.add(VectorField(func))

    .. manim:: SizingAndSpacing

        class SizingAndSpacing(Scene):
            def construct(self):
                func = lambda pos: np.sin(pos[0]/2)*UR+np.cos(pos[1]/2)*LEFT
                vf = VectorField(func, delta_x=1)
                self.add(vf)
                self.wait()

                length_func = lambda x: x / 3
                vf2 = VectorField(func, delta_x=1, length_func=length_func)
                self.play(vf.animate.become(vf2))
                self.wait()

    .. manim:: ColoringVectorFields
        :save_last_frame:

        class ColoringVectorFields(Scene):
            def construct(self):
                func = lambda pos: pos-LEFT*5
                colors = [RED, YELLOW, BLUE, DARKER_GRAY]
                min_radius = Circle(radius=2,  color=colors[0]).shift(LEFT*5)
                max_radius = Circle(radius=10, color=colors[1]).shift(LEFT*5)
                vf = VectorField(func, min_magnitude=2, max_magnitude=10, colors=colors)
                self.add(vf, min_radius, max_radius)


    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        # Determining Vector positions:
        x_min: Optional[float] = -(config["frame_width"] + 1) / 2,
        x_max: Optional[float] = (config["frame_width"] + 1) / 2,
        y_min: Optional[float] = -(config["frame_height"] + 1) / 2,
        y_max: Optional[float] = (config["frame_height"] + 1) / 2,
        delta_x: float = 0.5,
        delta_y: float = 0.5,
        # Determining Vector appearance:
        min_magnitude: float = 0,
        max_magnitude: float = 2,
        colors: Sequence = DEFAULT_SCALAR_FIELD_COLORS,
        # Takes in actual norm, spits out displayed norm
        length_func: Callable[[float], float] = lambda norm: 0.45 * sigmoid(norm),
        opacity: float = 1.0,
        vector_config: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(func, **kwargs)
        # Rounding min and max values to fit delta value
        self.x_min = floor(x_min / delta_x) * delta_x
        self.x_max = ceil(x_max / delta_x) * delta_x
        self.y_min = floor(y_min / delta_y) * delta_y
        self.y_max = ceil(y_max / delta_y) * delta_y
        self.delta_x = delta_x
        self.delta_y = delta_y

        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.colors = colors
        self.length_func = length_func
        self.opacity = opacity
        if vector_config is None:
            vector_config = {}
        self.vector_config = vector_config
        self.func = func
        self.color_gradient = get_color_gradient_function(
            self.min_magnitude, self.max_magnitude, self.colors
        )

        x_range = np.arange(self.x_min, self.x_max, self.delta_x)
        y_range = np.arange(self.y_min, self.y_max, self.delta_y)
        for x, y in it.product(x_range, y_range):
            self.add(self.get_vector(x * RIGHT + y * UP))
        self.set_opacity(self.opacity)

    def get_vector(self, point: np.ndarray):
        """Creates a vector in the `VectorField`.

        The created vector is based on the function of the `VectorField` and is
        rooted in the given point. Color and length fit the specifications of
        this `VectorField`.

        Parameters
        ----------
        point
            The root point of the vector.
        kwargs : Any
            Additional arguments to be passed to the :class:`~.Vector`-constructor

        """
        output = np.array(self.func(point))
        norm = get_norm(output)
        if not norm == 0:
            output *= self.length_func(norm) / norm
        vect = Vector(output, **self.vector_config)
        vect.shift(point)
        vect.set_color(self.color_gradient(norm))
        return vect


class StreamLines(VGroup):
    """StreamLines represented a vector field by showing it's flow by using moving agents.

    `StreamLines` are allways based on a function defining the vector at every position.
    This the values of this functions is displayed as a grid of vectors.
    The color of each vector is determined by it's magnitude.
    A color gradient can be used to color the vectors in a defined interval of magnitudes.

    Parameters
    ----------
    func

    """

    def __init__(
        self,
        func,
        # Config for choosing start points
        x_min=-8,
        x_max=8,
        y_min=-5,
        y_max=5,
        delta_x=0.5,
        delta_y=0.5,
        n_repeats=1,
        noise_factor=None,
        # Config for drawing lines
        dt=0.05,
        virtual_time=3,
        n_anchors_per_line=100,
        stroke_width=1,
        stroke_color=WHITE,
        color_by_arc_length=True,
        # Min and max arc lengths meant to define
        # the color range, should color_by_arc_length be True
        min_arc_length=0,
        max_arc_length=12,
        color_by_magnitude=False,
        # Min and max magnitudes meant to define
        # the color range, should color_by_magnitude be True
        min_magnitude=0.5,
        max_magnitude=1.5,
        colors=DEFAULT_SCALAR_FIELD_COLORS,
        cutoff_norm=15,
        **kwargs
    ):
        VGroup.__init__(
            self, stroke_color=stroke_color, stroke_width=stroke_width, **kwargs
        )
        self.func = func
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.n_repeats = n_repeats
        self.noise_factor = noise_factor
        self.dt = dt
        self.virtual_time = virtual_time
        self.n_anchors_per_line = n_anchors_per_line
        self.color_by_arc_length = color_by_arc_length
        self.min_arc_length = min_arc_length
        self.max_arc_length = max_arc_length
        self.color_by_magnitude = color_by_magnitude
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.colors = colors
        self.cutoff_norm = cutoff_norm

        start_points = self.get_start_points()
        for point in start_points:
            points = [point]
            for _ in np.arange(0, self.virtual_time, dt):
                last_point = points[-1]
                points.append(last_point + dt * func(last_point))
                if get_norm(last_point) > self.cutoff_norm:
                    break
            line = VMobject()
            step = max(1, int(len(points) / self.n_anchors_per_line))
            line.set_points_smoothly(points[::step])
            self.add(line)

        self.set_stroke(self.stroke_color, self.stroke_width)

        if self.color_by_arc_length:
            len_to_rgb = get_color_gradient_function(
                self.min_arc_length,
                self.max_arc_length,
                colors=self.colors,
            )
            for line in self:
                arc_length = line.get_arc_length()
                rgb = len_to_rgb([arc_length])[0]
                color = rgb_to_color(rgb)
                line.set_color(color)
        elif self.color_by_magnitude:
            image_file = get_color_field_image_file(
                lambda p: get_norm(func(p)),
                min_value=self.min_magnitude,
                max_value=self.max_magnitude,
                colors=self.colors,
            )
            self.color_using_background_image(image_file)

    def get_start_points(self):
        x_min = self.x_min
        x_max = self.x_max
        y_min = self.y_min
        y_max = self.y_max
        delta_x = self.delta_x
        delta_y = self.delta_y
        n_repeats = self.n_repeats
        noise_factor = self.noise_factor

        if noise_factor is None:
            noise_factor = delta_y / 2
        return np.array(
            [
                x * RIGHT + y * UP + noise_factor * np.random.random(3)
                for n in range(n_repeats)
                for x in np.arange(x_min, x_max + delta_x, delta_x)
                for y in np.arange(y_min, y_max + delta_y, delta_y)
            ]
        )


# TODO: Make it so that you can have a group of stream_lines
# varying in response to a changing vector field, and still
# animate the resulting flow
class ShowPassingFlashWithThinningStrokeWidth(AnimationGroup):
    def __init__(self, vmobject, n_segments=10, time_width=0.1, remover=True, **kwargs):
        self.n_segments = n_segments
        self.time_width = time_width
        self.remover = remover
        max_stroke_width = vmobject.get_stroke_width()
        max_time_width = kwargs.pop("time_width", self.time_width)
        AnimationGroup.__init__(
            self,
            *[
                ShowPassingFlash(
                    vmobject.deepcopy().set_stroke(width=stroke_width),
                    time_width=time_width,
                    **kwargs,
                )
                for stroke_width, time_width in zip(
                    np.linspace(0, max_stroke_width, self.n_segments),
                    np.linspace(max_time_width, 0, self.n_segments),
                )
            ],
        )


# TODO, this is untested after turning it from a
# ContinualAnimation into a VGroup
class AnimatedStreamLines(VGroup):
    def __init__(
        self,
        stream_lines,
        lag_range=4,
        line_anim_class=ShowPassingFlash,
        line_anim_config={
            "run_time": 4,
            "rate_func": linear,
            "time_width": 0.3,
        },
        **kwargs
    ):
        VGroup.__init__(self, **kwargs)
        self.stream_lines = stream_lines
        self.lag_range = lag_range
        self.line_anim_class = line_anim_class
        self.line_anim_config = line_anim_config
        for line in stream_lines:
            line.anim = self.line_anim_class(line, **self.line_anim_config)
            line.anim.begin()
            line.time = -self.lag_range * random.random()
            self.add(line.anim.mobject)

        self.add_updater(lambda m, dt: m.update(dt))

    def update(self, dt):
        stream_lines = self.stream_lines
        for line in stream_lines:
            line.time += dt
            adjusted_time = max(line.time, 0) % line.anim.run_time
            line.anim.interpolate(adjusted_time / line.anim.run_time)
