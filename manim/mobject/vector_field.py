"""Mobjects representing vector fields."""

from __future__ import annotations

__all__ = [
    "VectorField",
    "ArrowVectorField",
    "StreamLines",
]

import itertools as it
import random
from math import ceil, floor
from typing import Callable, Iterable, Sequence

import numpy as np
from colour import Color
from PIL import Image

from manim.animation.updaters.update import UpdateFromAlphaFunc
from manim.mobject.geometry.line import Vector
from manim.mobject.graphing.coordinate_systems import CoordinateSystem

from .. import config
from ..animation.composition import AnimationGroup, Succession
from ..animation.creation import Create
from ..animation.indication import ShowPassingFlash
from ..constants import OUT, RIGHT, UP, RendererType
from ..mobject.mobject import Mobject
from ..mobject.types.vectorized_mobject import VGroup
from ..mobject.utils import get_vectorized_mobject_class
from ..utils.bezier import interpolate, inverse_interpolate
from ..utils.color import BLUE_E, GREEN, RED, YELLOW, color_to_rgb, rgb_to_color
from ..utils.rate_functions import ease_out_sine, linear
from ..utils.simple_functions import sigmoid

DEFAULT_SCALAR_FIELD_COLORS: list = [BLUE_E, GREEN, YELLOW, RED]


class VectorField(VGroup):
    """A vector field.

    Vector fields are based on a function defining a vector at every position.
    This class does by default not include any visible elements but provides
    methods to move other :class:`~.Mobject` s along the vector field.

    Parameters
    ----------
    func
        The function defining the rate of change at every position of the `VectorField`.
    color
        The color of the vector field. If set, position-specific coloring is disabled.
    color_scheme
        A function mapping a vector to a single value. This value gives the position in the color gradient defined using `min_color_scheme_value`, `max_color_scheme_value` and `colors`.
    min_color_scheme_value
        The value of the color_scheme function to be mapped to the first color in `colors`. Lower values also result in the first color of the gradient.
    max_color_scheme_value
        The value of the color_scheme function to be mapped to the last color in `colors`. Higher values also result in the last color of the gradient.
    colors
        The colors defining the color gradient of the vector field.
    kwargs
        Additional arguments to be passed to the :class:`~.VGroup` constructor

    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        color: Color | None = None,
        color_scheme: Callable[[np.ndarray], float] | None = None,
        min_color_scheme_value: float = 0,
        max_color_scheme_value: float = 2,
        colors: Sequence[Color] = DEFAULT_SCALAR_FIELD_COLORS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.func = func
        if color is None:
            self.single_color = False
            if color_scheme is None:

                def color_scheme(p):
                    return np.linalg.norm(p)

            self.color_scheme = color_scheme  # TODO maybe other default for direction?
            self.rgbs = np.array(list(map(color_to_rgb, colors)))

            def pos_to_rgb(pos: np.ndarray) -> tuple[float, float, float, float]:
                vec = self.func(pos)
                color_value = np.clip(
                    self.color_scheme(vec),
                    min_color_scheme_value,
                    max_color_scheme_value,
                )
                alpha = inverse_interpolate(
                    min_color_scheme_value,
                    max_color_scheme_value,
                    color_value,
                )
                alpha *= len(self.rgbs) - 1
                c1 = self.rgbs[int(alpha)]
                c2 = self.rgbs[min(int(alpha + 1), len(self.rgbs) - 1)]
                alpha %= 1
                return interpolate(c1, c2, alpha)

            self.pos_to_rgb = pos_to_rgb
            self.pos_to_color = lambda pos: rgb_to_color(self.pos_to_rgb(pos))
        else:
            self.single_color = True
            self.color = color
        self.submob_movement_updater = None

    @staticmethod
    def shift_func(
        func: Callable[[np.ndarray], np.ndarray],
        shift_vector: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Shift a vector field function.

        Parameters
        ----------
        func
            The function defining a vector field.
        shift_vector
            The shift to be applied to the vector field.

        Returns
        -------
        `Callable[[np.ndarray], np.ndarray]`
            The shifted vector field function.

        """
        return lambda p: func(p - shift_vector)

    @staticmethod
    def scale_func(
        func: Callable[[np.ndarray], np.ndarray],
        scalar: float,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Scale a vector field function.

        Parameters
        ----------
        func
            The function defining a vector field.
        scalar
            The scalar to be applied to the vector field.

        Examples
        --------
        .. manim:: ScaleVectorFieldFunction

            class ScaleVectorFieldFunction(Scene):
                def construct(self):
                    func = lambda pos: np.sin(pos[1]) * RIGHT + np.cos(pos[0]) * UP
                    vector_field = ArrowVectorField(func)
                    self.add(vector_field)
                    self.wait()

                    func = VectorField.scale_func(func, 0.5)
                    self.play(vector_field.animate.become(ArrowVectorField(func)))
                    self.wait()

        Returns
        -------
        `Callable[[np.ndarray], np.ndarray]`
            The scaled vector field function.

        """
        return lambda p: func(p * scalar)

    def fit_to_coordinate_system(self, coordinate_system: CoordinateSystem):
        """Scale the vector field to fit a coordinate system.

        This method is useful when the vector field is defined in a coordinate system
        different from the one used to display the vector field.

        This method can only be used once because it transforms the origin of each vector.

        Parameters
        ----------
        coordinate_system
            The coordinate system to fit the vector field to.

        """
        self.apply_function(lambda pos: coordinate_system.coords_to_point(*pos))

    def nudge(
        self,
        mob: Mobject,
        dt: float = 1,
        substeps: int = 1,
        pointwise: bool = False,
    ) -> VectorField:
        """Nudge a :class:`~.Mobject` along the vector field.

        Parameters
        ----------
        mob
            The mobject to move along the vector field
        dt
            A scalar to the amount the mobject is moved along the vector field.
            The actual distance is based on the magnitude of the vector field.
        substeps
            The amount of steps the whole nudge is divided into. Higher values
            give more accurate approximations.
        pointwise
            Whether to move the mobject along the vector field. If `False` the
            vector field takes effect on the center of the given
            :class:`~.Mobject`. If `True` the vector field takes effect on the
            points of the individual points of the :class:`~.Mobject`,
            potentially distorting it.

        Returns
        -------
        VectorField
            This vector field.

        Examples
        --------

        .. manim:: Nudging

            class Nudging(Scene):
                def construct(self):
                    func = lambda pos: np.sin(pos[1] / 2) * RIGHT + np.cos(pos[0] / 2) * UP
                    vector_field = ArrowVectorField(
                        func, x_range=[-7, 7, 1], y_range=[-4, 4, 1], length_func=lambda x: x / 2
                    )
                    self.add(vector_field)
                    circle = Circle(radius=2).shift(LEFT)
                    self.add(circle.copy().set_color(GRAY))
                    dot = Dot().move_to(circle)

                    vector_field.nudge(circle, -2, 60, True)
                    vector_field.nudge(dot, -2, 60)

                    circle.add_updater(vector_field.get_nudge_updater(pointwise=True))
                    dot.add_updater(vector_field.get_nudge_updater())
                    self.add(circle, dot)
                    self.wait(6)

        """

        def runge_kutta(self, p: Sequence[float], step_size: float) -> float:
            """Returns the change in position of a point along a vector field.
            Parameters
            ----------
            p
               The position of each point being moved along the vector field.
            step_size
               A scalar that is used to determine how much a point is shifted in a single step.

            Returns
            -------
            float
               How much the point is shifted.
            """
            k_1 = self.func(p)
            k_2 = self.func(p + step_size * (k_1 * 0.5))
            k_3 = self.func(p + step_size * (k_2 * 0.5))
            k_4 = self.func(p + step_size * k_3)
            return step_size / 6.0 * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)

        step_size = dt / substeps
        for _ in range(substeps):
            if pointwise:
                mob.apply_function(lambda p: p + runge_kutta(self, p, step_size))
            else:
                mob.shift(runge_kutta(self, mob.get_center(), step_size))
        return self

    def nudge_submobjects(
        self,
        dt: float = 1,
        substeps: int = 1,
        pointwise: bool = False,
    ) -> VectorField:
        """Apply a nudge along the vector field to all submobjects.

        Parameters
        ----------
        dt
            A scalar to the amount the mobject is moved along the vector field.
            The actual distance is based on the magnitude of the vector field.
        substeps
            The amount of steps the whole nudge is divided into. Higher values
            give more accurate approximations.
        pointwise
            Whether to move the mobject along the vector field. See :meth:`nudge` for details.

        Returns
        -------
        VectorField
            This vector field.

        """
        for mob in self.submobjects:
            self.nudge(mob, dt, substeps, pointwise)
        return self

    def get_nudge_updater(
        self,
        speed: float = 1,
        pointwise: bool = False,
    ) -> Callable[[Mobject, float], Mobject]:
        """Get an update function to move a :class:`~.Mobject` along the vector field.

        When used with :meth:`~.Mobject.add_updater`, the mobject will move along the vector field, where its speed is determined by the magnitude of the vector field.

        Parameters
        ----------
        speed
            At `speed=1` the distance a mobject moves per second is equal to the magnitude of the vector field along its path. The speed value scales the speed of such a mobject.
        pointwise
            Whether to move the mobject along the vector field. See :meth:`nudge` for details.

        Returns
        -------
        Callable[[Mobject, float], Mobject]
            The update function.
        """
        return lambda mob, dt: self.nudge(mob, dt * speed, pointwise=pointwise)

    def start_submobject_movement(
        self,
        speed: float = 1,
        pointwise: bool = False,
    ) -> VectorField:
        """Start continuously moving all submobjects along the vector field.

        Calling this method multiple times will result in removing the previous updater created by this method.

        Parameters
        ----------
        speed
            The speed at which to move the submobjects. See :meth:`get_nudge_updater` for details.
        pointwise
            Whether to move the mobject along the vector field. See :meth:`nudge` for details.

        Returns
        -------
        VectorField
            This vector field.

        """

        self.stop_submobject_movement()
        self.submob_movement_updater = lambda mob, dt: mob.nudge_submobjects(
            dt * speed,
            pointwise=pointwise,
        )
        self.add_updater(self.submob_movement_updater)
        return self

    def stop_submobject_movement(self) -> VectorField:
        """Stops the continuous movement started using :meth:`start_submobject_movement`.

        Returns
        -------
        VectorField
            This vector field.
        """
        self.remove_updater(self.submob_movement_updater)
        self.submob_movement_updater = None
        return self

    def get_colored_background_image(self, sampling_rate: int = 5) -> Image.Image:
        """Generate an image that displays the vector field.

        The color at each position is calculated by passing the positing through a
        series of steps:
        Calculate the vector field function at that position, map that vector to a
        single value using `self.color_scheme` and finally generate a color from
        that value using the color gradient.

        Parameters
        ----------
        sampling_rate
            The stepsize at which pixels get included in the image. Lower values give
            more accurate results, but may take a long time to compute.

        Returns
        -------
        Image.Imgae
            The vector field image.
        """
        if self.single_color:
            raise ValueError(
                "There is no point in generating an image if the vector field uses a single color.",
            )
        ph = int(config["pixel_height"] / sampling_rate)
        pw = int(config["pixel_width"] / sampling_rate)
        fw = config["frame_width"]
        fh = config["frame_height"]
        points_array = np.zeros((ph, pw, 3))
        x_array = np.linspace(-fw / 2, fw / 2, pw)
        y_array = np.linspace(fh / 2, -fh / 2, ph)
        x_array = x_array.reshape((1, len(x_array)))
        y_array = y_array.reshape((len(y_array), 1))
        x_array = x_array.repeat(ph, axis=0)
        y_array.repeat(pw, axis=1)  # TODO why not y_array = y_array.repeat(...)?
        points_array[:, :, 0] = x_array
        points_array[:, :, 1] = y_array
        rgbs = np.apply_along_axis(self.pos_to_rgb, 2, points_array)
        return Image.fromarray((rgbs * 255).astype("uint8"))

    def get_vectorized_rgba_gradient_function(
        self,
        start: float,
        end: float,
        colors: Iterable,
    ):
        """
        Generates a gradient of rgbas as a numpy array

        Parameters
        ----------
        start
            start value used for inverse interpolation at :func:`~.inverse_interpolate`
        end
            end value used for inverse interpolation at :func:`~.inverse_interpolate`
        colors
            list of colors to generate the gradient

        Returns
        -------
            function to generate the gradients as numpy arrays representing rgba values
        """
        rgbs = np.array([color_to_rgb(c) for c in colors])

        def func(values, opacity=1):
            alphas = inverse_interpolate(start, end, np.array(values))
            alphas = np.clip(alphas, 0, 1)
            scaled_alphas = alphas * (len(rgbs) - 1)
            indices = scaled_alphas.astype(int)
            next_indices = np.clip(indices + 1, 0, len(rgbs) - 1)
            inter_alphas = scaled_alphas % 1
            inter_alphas = inter_alphas.repeat(3).reshape((len(indices), 3))
            result = interpolate(rgbs[indices], rgbs[next_indices], inter_alphas)
            result = np.concatenate(
                (result, np.full([len(result), 1], opacity)),
                axis=1,
            )
            return result

        return func


class ArrowVectorField(VectorField):
    """A :class:`VectorField` represented by a set of change vectors.

    Vector fields are always based on a function defining the :class:`~.Vector` at every position.
    The values of this functions is displayed as a grid of vectors.
    By default the color of each vector is determined by it's magnitude.
    Other color schemes can be used however.

    Parameters
    ----------
    func
        The function defining the rate of change at every position of the vector field.
    color
        The color of the vector field. If set, position-specific coloring is disabled.
    color_scheme
        A function mapping a vector to a single value. This value gives the position in the color gradient defined using `min_color_scheme_value`, `max_color_scheme_value` and `colors`.
    min_color_scheme_value
        The value of the color_scheme function to be mapped to the first color in `colors`. Lower values also result in the first color of the gradient.
    max_color_scheme_value
        The value of the color_scheme function to be mapped to the last color in `colors`. Higher values also result in the last color of the gradient.
    colors
        The colors defining the color gradient of the vector field.
    x_range
        A sequence of x_min, x_max, delta_x
    y_range
        A sequence of y_min, y_max, delta_y
    z_range
        A sequence of z_min, z_max, delta_z
    three_dimensions
        Enables three_dimensions. Default set to False, automatically turns True if
        z_range is not None.
    length_func
        The function determining the displayed size of the vectors. The actual size
        of the vector is passed, the returned value will be used as display size for the
        vector. By default this is used to cap the displayed size of vectors to reduce the clutter.
    opacity
        The opacity of the arrows.
    vector_config
        Additional arguments to be passed to the :class:`~.Vector` constructor
    kwargs
        Additional arguments to be passed to the :class:`~.VGroup` constructor

    Examples
    --------

    .. manim:: BasicUsage
        :save_last_frame:

        class BasicUsage(Scene):
            def construct(self):
                func = lambda pos: ((pos[0] * UR + pos[1] * LEFT) - pos) / 3
                self.add(ArrowVectorField(func))

    .. manim:: SizingAndSpacing

        class SizingAndSpacing(Scene):
            def construct(self):
                func = lambda pos: np.sin(pos[0] / 2) * UR + np.cos(pos[1] / 2) * LEFT
                vf = ArrowVectorField(func, x_range=[-7, 7, 1])
                self.add(vf)
                self.wait()

                length_func = lambda x: x / 3
                vf2 = ArrowVectorField(func, x_range=[-7, 7, 1], length_func=length_func)
                self.play(vf.animate.become(vf2))
                self.wait()

    .. manim:: Coloring
        :save_last_frame:

        class Coloring(Scene):
            def construct(self):
                func = lambda pos: pos - LEFT * 5
                colors = [RED, YELLOW, BLUE, DARK_GRAY]
                min_radius = Circle(radius=2, color=colors[0]).shift(LEFT * 5)
                max_radius = Circle(radius=10, color=colors[-1]).shift(LEFT * 5)
                vf = ArrowVectorField(
                    func, min_color_scheme_value=2, max_color_scheme_value=10, colors=colors
                )
                self.add(vf, min_radius, max_radius)

    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        color: Color | None = None,
        color_scheme: Callable[[np.ndarray], float] | None = None,
        min_color_scheme_value: float = 0,
        max_color_scheme_value: float = 2,
        colors: Sequence[Color] = DEFAULT_SCALAR_FIELD_COLORS,
        # Determining Vector positions:
        x_range: Sequence[float] = None,
        y_range: Sequence[float] = None,
        z_range: Sequence[float] = None,
        three_dimensions: bool = False,  # Automatically True if z_range is set
        # Takes in actual norm, spits out displayed norm
        length_func: Callable[[float], float] = lambda norm: 0.45 * sigmoid(norm),
        opacity: float = 1.0,
        vector_config: dict | None = None,
        **kwargs,
    ):
        self.x_range = x_range or [
            floor(-config["frame_width"] / 2),
            ceil(config["frame_width"] / 2),
        ]
        self.y_range = y_range or [
            floor(-config["frame_height"] / 2),
            ceil(config["frame_height"] / 2),
        ]
        self.ranges = [self.x_range, self.y_range]

        if three_dimensions or z_range:
            self.z_range = z_range or self.y_range.copy()
            self.ranges += [self.z_range]
        else:
            self.ranges += [[0, 0]]

        for i in range(len(self.ranges)):
            if len(self.ranges[i]) == 2:
                self.ranges[i] += [0.5]
            self.ranges[i][1] += self.ranges[i][2]

        self.x_range, self.y_range, self.z_range = self.ranges

        super().__init__(
            func,
            color,
            color_scheme,
            min_color_scheme_value,
            max_color_scheme_value,
            colors,
            **kwargs,
        )

        self.length_func = length_func
        self.opacity = opacity
        if vector_config is None:
            vector_config = {}
        self.vector_config = vector_config
        self.func = func

        x_range = np.arange(*self.x_range)
        y_range = np.arange(*self.y_range)
        z_range = np.arange(*self.z_range)
        self.add(
            *[
                self.get_vector(x * RIGHT + y * UP + z * OUT)
                for x, y, z in it.product(x_range, y_range, z_range)
            ]
        )
        self.set_opacity(self.opacity)

    def get_vector(self, point: np.ndarray):
        """Creates a vector in the vector field.

        The created vector is based on the function of the vector field and is
        rooted in the given point. Color and length fit the specifications of
        this vector field.

        Parameters
        ----------
        point
            The root point of the vector.

        """
        output = np.asarray(self.func(point))
        norm = np.linalg.norm(output)
        if norm != 0:
            output *= self.length_func(norm) / norm
        vect = Vector(output, **self.vector_config)
        vect.shift(point)
        if self.single_color:
            vect.set_color(self.color)
        else:
            vect.set_color(self.pos_to_color(point))
        return vect


class StreamLines(VectorField):
    """StreamLines represent the flow of a :class:`VectorField` using the trace of moving agents.

    Vector fields are always based on a function defining the vector at every position.
    The values of this functions is displayed by moving many agents along the vector field
    and showing their trace.

    Parameters
    ----------
    func
        The function defining the rate of change at every position of the vector field.
    color
        The color of the vector field. If set, position-specific coloring is disabled.
    color_scheme
        A function mapping a vector to a single value. This value gives the position in the color gradient defined using `min_color_scheme_value`, `max_color_scheme_value` and `colors`.
    min_color_scheme_value
        The value of the color_scheme function to be mapped to the first color in `colors`. Lower values also result in the first color of the gradient.
    max_color_scheme_value
        The value of the color_scheme function to be mapped to the last color in `colors`. Higher values also result in the last color of the gradient.
    colors
        The colors defining the color gradient of the vector field.
    x_range
        A sequence of x_min, x_max, delta_x
    y_range
        A sequence of y_min, y_max, delta_y
    z_range
        A sequence of z_min, z_max, delta_z
    three_dimensions
        Enables three_dimensions. Default set to False, automatically turns True if
        z_range is not None.
    noise_factor
        The amount by which the starting position of each agent is altered along each axis. Defaults to :code:`delta_y / 2` if not defined.
    n_repeats
        The number of agents generated at each starting point.
    dt
        The factor by which the distance an agent moves per step is stretched. Lower values result in a better approximation of the trajectories in the vector field.
    virtual_time
        The time the agents get to move in the vector field. Higher values therefore result in longer stream lines. However, this whole time gets simulated upon creation.
    max_anchors_per_line
        The maximum number of anchors per line. Lines with more anchors get reduced in complexity, not in length.
    padding
        The distance agents can move out of the generation area before being terminated.
    stroke_width
        The stroke with of the stream lines.
    opacity
        The opacity of the stream lines.

    Examples
    --------

    .. manim:: BasicUsage
        :save_last_frame:

        class BasicUsage(Scene):
            def construct(self):
                func = lambda pos: ((pos[0] * UR + pos[1] * LEFT) - pos) / 3
                self.add(StreamLines(func))

    .. manim:: SpawningAndFlowingArea
        :save_last_frame:

        class SpawningAndFlowingArea(Scene):
            def construct(self):
                func = lambda pos: np.sin(pos[0]) * UR + np.cos(pos[1]) * LEFT + pos / 5
                stream_lines = StreamLines(
                    func, x_range=[-3, 3, 0.2], y_range=[-2, 2, 0.2], padding=1
                )

                spawning_area = Rectangle(width=6, height=4)
                flowing_area = Rectangle(width=8, height=6)
                labels = [Tex("Spawning Area"), Tex("Flowing Area").shift(DOWN * 2.5)]
                for lbl in labels:
                    lbl.add_background_rectangle(opacity=0.6, buff=0.05)

                self.add(stream_lines, spawning_area, flowing_area, *labels)

    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        color: Color | None = None,
        color_scheme: Callable[[np.ndarray], float] | None = None,
        min_color_scheme_value: float = 0,
        max_color_scheme_value: float = 2,
        colors: Sequence[Color] = DEFAULT_SCALAR_FIELD_COLORS,
        # Determining stream line starting positions:
        x_range: Sequence[float] = None,
        y_range: Sequence[float] = None,
        z_range: Sequence[float] = None,
        three_dimensions: bool = False,
        noise_factor: float | None = None,
        n_repeats=1,
        # Determining how lines are drawn
        dt=0.05,
        virtual_time=3,
        max_anchors_per_line=100,
        padding=3,
        # Determining stream line appearance:
        stroke_width=1,
        opacity=1,
        **kwargs,
    ):
        self.x_range = x_range or [
            floor(-config["frame_width"] / 2),
            ceil(config["frame_width"] / 2),
        ]
        self.y_range = y_range or [
            floor(-config["frame_height"] / 2),
            ceil(config["frame_height"] / 2),
        ]
        self.ranges = [self.x_range, self.y_range]

        if three_dimensions or z_range:
            self.z_range = z_range or self.y_range.copy()
            self.ranges += [self.z_range]
        else:
            self.ranges += [[0, 0]]

        for i in range(len(self.ranges)):
            if len(self.ranges[i]) == 2:
                self.ranges[i] += [0.5]
            self.ranges[i][1] += self.ranges[i][2]

        self.x_range, self.y_range, self.z_range = self.ranges

        super().__init__(
            func,
            color,
            color_scheme,
            min_color_scheme_value,
            max_color_scheme_value,
            colors,
            **kwargs,
        )

        self.noise_factor = (
            noise_factor if noise_factor is not None else self.y_range[2] / 2
        )
        self.n_repeats = n_repeats
        self.virtual_time = virtual_time
        self.max_anchors_per_line = max_anchors_per_line
        self.padding = padding
        self.stroke_width = stroke_width

        half_noise = self.noise_factor / 2
        np.random.seed(0)
        start_points = np.array(
            [
                (x - half_noise) * RIGHT
                + (y - half_noise) * UP
                + (z - half_noise) * OUT
                + self.noise_factor * np.random.random(3)
                for n in range(self.n_repeats)
                for x in np.arange(*self.x_range)
                for y in np.arange(*self.y_range)
                for z in np.arange(*self.z_range)
            ],
        )

        def outside_box(p):
            return (
                p[0] < self.x_range[0] - self.padding
                or p[0] > self.x_range[1] + self.padding - self.x_range[2]
                or p[1] < self.y_range[0] - self.padding
                or p[1] > self.y_range[1] + self.padding - self.y_range[2]
                or p[2] < self.z_range[0] - self.padding
                or p[2] > self.z_range[1] + self.padding - self.z_range[2]
            )

        max_steps = ceil(virtual_time / dt) + 1
        if not self.single_color:
            self.background_img = self.get_colored_background_image()
            if config["renderer"] == RendererType.OPENGL:
                self.values_to_rgbas = self.get_vectorized_rgba_gradient_function(
                    min_color_scheme_value,
                    max_color_scheme_value,
                    colors,
                )
        for point in start_points:
            points = [point]
            for _ in range(max_steps):
                last_point = points[-1]
                new_point = last_point + dt * func(last_point)
                if outside_box(new_point):
                    break
                points.append(new_point)
            step = max_steps
            if not step:
                continue
            line = get_vectorized_mobject_class()()
            line.duration = step * dt
            step = max(1, int(len(points) / self.max_anchors_per_line))
            line.set_points_smoothly(points[::step])
            if self.single_color:
                line.set_stroke(self.color)
            else:
                if config.renderer == RendererType.OPENGL:
                    # scaled for compatibility with cairo
                    line.set_stroke(width=self.stroke_width / 4.0)
                    norms = np.array(
                        [np.linalg.norm(self.func(point)) for point in line.points],
                    )
                    line.set_rgba_array_direct(
                        self.values_to_rgbas(norms, opacity),
                        name="stroke_rgba",
                    )
                else:
                    if np.any(self.z_range != np.array([0, 0.5, 0.5])):
                        line.set_stroke(
                            [self.pos_to_color(p) for p in line.get_anchors()],
                        )
                    else:
                        line.color_using_background_image(self.background_img)
                    line.set_stroke(width=self.stroke_width, opacity=opacity)
            self.add(line)
        self.stream_lines = [*self.submobjects]

    def create(
        self,
        lag_ratio: float | None = None,
        run_time: Callable[[float], float] | None = None,
        **kwargs,
    ) -> AnimationGroup:
        """The creation animation of the stream lines.

        The stream lines appear in random order.

        Parameters
        ----------
        lag_ratio
            The lag ratio of the animation.
            If undefined, it will be selected so that the total animation length is 1.5 times the run time of each stream line creation.
        run_time
            The run time of every single stream line creation. The runtime of the whole animation might be longer due to the `lag_ratio`.
            If undefined, the virtual time of the stream lines is used as run time.

        Returns
        -------
        :class:`~.AnimationGroup`
            The creation animation of the stream lines.

        Examples
        --------

        .. manim:: StreamLineCreation

            class StreamLineCreation(Scene):
                def construct(self):
                    func = lambda pos: (pos[0] * UR + pos[1] * LEFT) - pos
                    stream_lines = StreamLines(
                        func,
                        color=YELLOW,
                        x_range=[-7, 7, 1],
                        y_range=[-4, 4, 1],
                        stroke_width=3,
                        virtual_time=1,  # use shorter lines
                        max_anchors_per_line=5,  # better performance with fewer anchors
                    )
                    self.play(stream_lines.create())  # uses virtual_time as run_time
                    self.wait()

        """
        if run_time is None:
            run_time = self.virtual_time
        if lag_ratio is None:
            lag_ratio = run_time / 2 / len(self.submobjects)

        animations = [
            Create(line, run_time=run_time, **kwargs) for line in self.stream_lines
        ]
        random.shuffle(animations)
        return AnimationGroup(*animations, lag_ratio=lag_ratio)

    def start_animation(
        self,
        warm_up: bool = True,
        flow_speed: float = 1,
        time_width: float = 0.3,
        rate_func: Callable[[float], float] = linear,
        line_animation_class: type[ShowPassingFlash] = ShowPassingFlash,
        **kwargs,
    ) -> None:
        """Animates the stream lines using an updater.

        The stream lines will continuously flow

        Parameters
        ----------
        warm_up
            If `True` the animation is initialized line by line. Otherwise it starts with all lines shown.
        flow_speed
            At `flow_speed=1` the distance the flow moves per second is equal to the magnitude of the vector field along its path. The speed value scales the speed of this flow.
        time_width
            The proportion of the stream line shown while being animated
        rate_func
            The rate function of each stream line flashing
        line_animation_class
            The animation class being used

        Examples
        --------

        .. manim:: ContinuousMotion

            class ContinuousMotion(Scene):
                def construct(self):
                    func = lambda pos: np.sin(pos[0] / 2) * UR + np.cos(pos[1] / 2) * LEFT
                    stream_lines = StreamLines(func, stroke_width=3, max_anchors_per_line=30)
                    self.add(stream_lines)
                    stream_lines.start_animation(warm_up=False, flow_speed=1.5)
                    self.wait(stream_lines.virtual_time / stream_lines.flow_speed)

        """

        for line in self.stream_lines:
            run_time = line.duration / flow_speed
            line.anim = line_animation_class(
                line,
                run_time=run_time,
                rate_func=rate_func,
                time_width=time_width,
                **kwargs,
            )
            line.anim.begin()
            line.time = random.random() * self.virtual_time
            if warm_up:
                line.time *= -1
            self.add(line.anim.mobject)

        def updater(mob, dt):
            for line in mob.stream_lines:
                line.time += dt * flow_speed
                if line.time >= self.virtual_time:
                    line.time -= self.virtual_time
                line.anim.interpolate(np.clip(line.time / line.anim.run_time, 0, 1))

        self.add_updater(updater)
        self.flow_animation = updater
        self.flow_speed = flow_speed
        self.time_width = time_width

    def end_animation(self) -> AnimationGroup:
        """End the stream line animation smoothly.

        Returns an animation resulting in fully displayed stream lines without a noticeable cut.

        Returns
        -------
        :class:`~.AnimationGroup`
            The animation fading out the running stream animation.

        Raises
        ------
        ValueError
            if no stream line animation is running

        Examples
        --------

        .. manim:: EndAnimation

            class EndAnimation(Scene):
                def construct(self):
                    func = lambda pos: np.sin(pos[0] / 2) * UR + np.cos(pos[1] / 2) * LEFT
                    stream_lines = StreamLines(
                        func, stroke_width=3, max_anchors_per_line=5, virtual_time=1, color=BLUE
                    )
                    self.add(stream_lines)
                    stream_lines.start_animation(warm_up=False, flow_speed=1.5, time_width=0.5)
                    self.wait(1)
                    self.play(stream_lines.end_animation())

        """

        if self.flow_animation is None:
            raise ValueError("You have to start the animation before fading it out.")

        def hide_and_wait(mob, alpha):
            if alpha == 0:
                mob.set_stroke(opacity=0)
            elif alpha == 1:
                mob.set_stroke(opacity=1)

        def finish_updater_cycle(line, alpha):
            line.time += dt * self.flow_speed
            line.anim.interpolate(min(line.time / line.anim.run_time, 1))
            if alpha == 1:
                self.remove(line.anim.mobject)
                line.anim.finish()

        max_run_time = self.virtual_time / self.flow_speed
        creation_rate_func = ease_out_sine
        creation_staring_speed = creation_rate_func(0.001) * 1000
        creation_run_time = (
            max_run_time / (1 + self.time_width) * creation_staring_speed
        )
        # creation_run_time is calculated so that the creation animation starts at the same speed
        # as the regular line flash animation but eases out.

        dt = 1 / config["frame_rate"]
        animations = []
        self.remove_updater(self.flow_animation)
        self.flow_animation = None

        for line in self.stream_lines:
            create = Create(
                line,
                run_time=creation_run_time,
                rate_func=creation_rate_func,
            )
            if line.time <= 0:
                animations.append(
                    Succession(
                        UpdateFromAlphaFunc(
                            line,
                            hide_and_wait,
                            run_time=-line.time / self.flow_speed,
                        ),
                        create,
                    ),
                )
                self.remove(line.anim.mobject)
                line.anim.finish()
            else:
                remaining_time = max_run_time - line.time / self.flow_speed
                animations.append(
                    Succession(
                        UpdateFromAlphaFunc(
                            line,
                            finish_updater_cycle,
                            run_time=remaining_time,
                        ),
                        create,
                    ),
                )
        return AnimationGroup(*animations)


# TODO: Variant of StreamLines that is able to respond to changes in the vector field function
