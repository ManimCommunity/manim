"""Mobject representing a number line."""

from __future__ import annotations

from manim.mobject.mobject import Mobject
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject

__all__ = ["NumberLine", "UnitInterval"]


from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from manim.mobject.geometry.tips import ArrowTip
    from manim.typing import Point3DLike

import numpy as np

from manim import config
from manim.constants import *
from manim.mobject.geometry.line import Line
from manim.mobject.graphing.scale import LinearBase, _ScaleBase
from manim.mobject.text.numbers import DecimalNumber
from manim.mobject.text.tex_mobject import MathTex, Tex
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.utils.bezier import interpolate
from manim.utils.config_ops import merge_dicts_recursively
from manim.utils.space_ops import normalize


class NumberLine(Line):
    """Creates a number line with tick marks.

    Parameters
    ----------
    x_range
        The ``[x_min, x_max, x_step]`` values to create the line.
    length
        The length of the number line.
    unit_size
        The distance between each tick of the line. Overwritten by :attr:`length`, if specified.
    include_ticks
        Whether to include ticks on the number line.
    tick_size
        The length of each tick mark.
    numbers_with_elongated_ticks
        An iterable of specific values with elongated ticks.
    longer_tick_multiple
        Influences how many times larger elongated ticks are than regular ticks (2 = 2x).
    rotation
        The angle (in radians) at which the line is rotated.
    stroke_width
        The thickness of the line.
    include_tip
        Whether to add a tip to the end of the line.
    tip_width
        The width of the tip.
    tip_height
        The height of the tip.
    tip_shape
        The mobject class used to construct the tip, or ``None`` (the
        default) for the default arrow tip. Passed classes have to inherit
        from :class:`.ArrowTip`.
    include_numbers
        Whether to add numbers to the tick marks. The number of decimal places is determined
        by the step size, this default can be overridden by ``decimal_number_config``.
    scaling
        The way the ``x_range`` is value is scaled, i.e. :class:`~.LogBase` for a logarithmic numberline. Defaults to :class:`~.LinearBase`.
    font_size
        The size of the label mobjects. Defaults to 36.
    label_direction
        The specific position to which label mobjects are added on the line.
    label_constructor
        Determines the mobject class that will be used to construct the labels of the number line.
    line_to_number_buff
        The distance between the line and the label mobject.
    decimal_number_config
        Arguments that can be passed to :class:`~.numbers.DecimalNumber` to influence number mobjects.
    numbers_to_exclude
        An explicit iterable of numbers to not be added to the number line.
    numbers_to_include
        An explicit iterable of numbers to add to the number line
    kwargs
        Additional arguments to be passed to :class:`~.Line`.


    .. note::

        Number ranges that include both negative and positive values will be generated
        from the 0 point, and may not include a tick at the min / max
        values as the tick locations are dependent on the step size.

    Examples
    --------
    .. manim:: NumberLineExample
        :save_last_frame:

        class NumberLineExample(Scene):
            def construct(self):
                l0 = NumberLine(
                    x_range=[-10, 10, 2],
                    length=10,
                    color=BLUE,
                    include_numbers=True,
                    label_direction=UP,
                )

                l1 = NumberLine(
                    x_range=[-10, 10, 2],
                    unit_size=0.5,
                    numbers_with_elongated_ticks=[-2, 4],
                    include_numbers=True,
                    font_size=24,
                )
                num6 = l1.numbers[8]
                num6.set_color(RED)

                l2 = NumberLine(
                    x_range=[-2.5, 2.5 + 0.5, 0.5],
                    length=12,
                    decimal_number_config={"num_decimal_places": 2},
                    include_numbers=True,
                )

                l3 = NumberLine(
                    x_range=[-5, 5 + 1, 1],
                    length=6,
                    include_tip=True,
                    include_numbers=True,
                    rotation=10 * DEGREES,
                )

                line_group = VGroup(l0, l1, l2, l3).arrange(DOWN, buff=1)
                self.add(line_group)
    """

    def __init__(
        self,
        x_range: Sequence[float] | None = None,  # must be first
        length: float | None = None,
        unit_size: float = 1,
        # ticks
        include_ticks: bool = True,
        tick_size: float = 0.1,
        numbers_with_elongated_ticks: Iterable[float] | None = None,
        longer_tick_multiple: int = 2,
        exclude_origin_tick: bool = False,
        # visuals
        rotation: float = 0,
        stroke_width: float = 2.0,
        # tip
        include_tip: bool = False,
        tip_width: float = DEFAULT_ARROW_TIP_LENGTH,
        tip_height: float = DEFAULT_ARROW_TIP_LENGTH,
        tip_shape: type[ArrowTip] | None = None,
        # numbers/labels
        include_numbers: bool = False,
        font_size: float = 36,
        label_direction: Sequence[float] = DOWN,
        label_constructor: VMobject = MathTex,
        scaling: _ScaleBase = LinearBase(),
        line_to_number_buff: float = MED_SMALL_BUFF,
        decimal_number_config: dict | None = None,
        numbers_to_exclude: Iterable[float] | None = None,
        numbers_to_include: Iterable[float] | None = None,
        **kwargs,
    ):
        # avoid mutable arguments in defaults
        if numbers_to_exclude is None:
            numbers_to_exclude = []
        if numbers_with_elongated_ticks is None:
            numbers_with_elongated_ticks = []

        if x_range is None:
            x_range = [
                round(-config["frame_x_radius"]),
                round(config["frame_x_radius"]),
                1,
            ]
        elif len(x_range) == 2:
            # adds x_step if not specified. not sure how to feel about this. a user can't know default without peeking at source code
            x_range = [*x_range, 1]

        if decimal_number_config is None:
            decimal_number_config = {
                "num_decimal_places": self._decimal_places_from_step(x_range[2]),
            }

        # turn into a NumPy array to scale by just applying the function
        self.x_range = np.array(x_range, dtype=float)
        self.x_min, self.x_max, self.x_step = scaling.function(self.x_range)
        self.length = length
        self.unit_size = unit_size
        # ticks
        self.include_ticks = include_ticks
        self.tick_size = tick_size
        self.numbers_with_elongated_ticks = numbers_with_elongated_ticks
        self.longer_tick_multiple = longer_tick_multiple
        self.exclude_origin_tick = exclude_origin_tick
        # visuals
        self.rotation = rotation
        # tip
        self.include_tip = include_tip
        self.tip_width = tip_width
        self.tip_height = tip_height
        # numbers
        self.font_size = font_size
        self.include_numbers = include_numbers
        self.label_direction = label_direction
        self.label_constructor = label_constructor
        self.line_to_number_buff = line_to_number_buff
        self.decimal_number_config = decimal_number_config
        self.numbers_to_exclude = numbers_to_exclude
        self.numbers_to_include = numbers_to_include

        self.scaling = scaling
        super().__init__(
            self.x_range[0] * RIGHT,
            self.x_range[1] * RIGHT,
            stroke_width=stroke_width,
            **kwargs,
        )

        if self.length:
            self.set_length(self.length)
            self.unit_size = self.get_unit_size()
        else:
            self.scale(self.unit_size)

        self.center()

        if self.include_tip:
            self.add_tip(
                tip_length=self.tip_height,
                tip_width=self.tip_width,
                tip_shape=tip_shape,
            )
            self.tip.set_stroke(self.stroke_color, self.stroke_width)

        if self.include_ticks:
            self.add_ticks()

        self.rotate(self.rotation)
        if self.include_numbers or self.numbers_to_include is not None:
            if self.scaling.custom_labels:
                tick_range = self.get_tick_range()

                self.add_labels(
                    dict(
                        zip(
                            tick_range,
                            self.scaling.get_custom_labels(
                                tick_range,
                                unit_decimal_places=decimal_number_config[
                                    "num_decimal_places"
                                ],
                            ),
                        )
                    ),
                )

            else:
                self.add_numbers(
                    x_values=self.numbers_to_include,
                    excluding=self.numbers_to_exclude,
                    font_size=self.font_size,
                )

    def rotate_about_zero(self, angle: float, axis: Sequence[float] = OUT, **kwargs):
        return self.rotate_about_number(0, angle, axis, **kwargs)

    def rotate_about_number(
        self, number: float, angle: float, axis: Sequence[float] = OUT, **kwargs
    ):
        return self.rotate(angle, axis, about_point=self.n2p(number), **kwargs)

    def add_ticks(self):
        """Adds ticks to the number line. Ticks can be accessed after creation
        via ``self.ticks``.
        """
        ticks = VGroup()
        elongated_tick_size = self.tick_size * self.longer_tick_multiple
        elongated_tick_offsets = self.numbers_with_elongated_ticks - self.x_min
        for x in self.get_tick_range():
            size = self.tick_size
            if np.any(np.isclose(x - self.x_min, elongated_tick_offsets)):
                size = elongated_tick_size
            ticks.add(self.get_tick(x, size))
        self.add(ticks)
        self.ticks = ticks

    def get_tick(self, x: float, size: float | None = None) -> Line:
        """Generates a tick and positions it along the number line.

        Parameters
        ----------
        x
            The position of the tick.
        size
            The factor by which the tick is scaled.

        Returns
        -------
        :class:`~.Line`
            A positioned tick.
        """
        if size is None:
            size = self.tick_size
        result = Line(size * DOWN, size * UP)
        result.rotate(self.get_angle())
        result.move_to(self.number_to_point(x))
        result.match_style(self)
        return result

    def get_tick_marks(self) -> VGroup:
        return self.ticks

    def get_tick_range(self) -> np.ndarray:
        """Generates the range of values on which labels are plotted based on the
        ``x_range`` attribute of the number line.

        Returns
        -------
        np.ndarray
            A numpy array of floats represnting values along the number line.
        """
        x_min, x_max, x_step = self.x_range
        if not self.include_tip:
            x_max += 1e-6

        # Handle cases where min and max are both positive or both negative
        if x_min < x_max < 0 or x_max > x_min > 0:
            tick_range = np.arange(x_min, x_max, x_step)
        else:
            start_point = 0
            if self.exclude_origin_tick:
                start_point += x_step

            x_min_segment = np.arange(start_point, np.abs(x_min) + 1e-6, x_step) * -1
            x_max_segment = np.arange(start_point, x_max, x_step)

            tick_range = np.unique(np.concatenate((x_min_segment, x_max_segment)))

        return self.scaling.function(tick_range)

    def number_to_point(self, number: float | np.ndarray) -> np.ndarray:
        """Accepts a value along the number line and returns a point with
        respect to the scene.
        Equivalent to `NumberLine @ number`

        Parameters
        ----------
        number
            The value to be transformed into a coordinate. Or a list of values.

        Returns
        -------
        np.ndarray
            A point with respect to the scene's coordinate system. Or a list of points.

        Examples
        --------

            >>> from manim import NumberLine
            >>> number_line = NumberLine()
            >>> number_line.number_to_point(0)
            array([0., 0., 0.])
            >>> number_line.number_to_point(1)
            array([1., 0., 0.])
            >>> number_line @ 1
            array([1., 0., 0.])
            >>> number_line.number_to_point([1, 2, 3])
            array([[1., 0., 0.],
                   [2., 0., 0.],
                   [3., 0., 0.]])
        """
        number = np.asarray(number)
        scalar = number.ndim == 0
        number = self.scaling.inverse_function(number)
        alphas = (number - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        alphas = float(alphas) if scalar else np.vstack(alphas)
        val = interpolate(self.get_start(), self.get_end(), alphas)
        return val

    def point_to_number(self, point: Sequence[float]) -> float:
        """Accepts a point with respect to the scene and returns
        a float along the number line.

        Parameters
        ----------
        point
            A sequence of values consisting of ``(x_coord, y_coord, z_coord)``.

        Returns
        -------
        float
            A float representing a value along the number line.

        Examples
        --------

            >>> from manim import NumberLine
            >>> number_line = NumberLine()
            >>> number_line.point_to_number((0, 0, 0))
            np.float64(0.0)
            >>> number_line.point_to_number((1, 0, 0))
            np.float64(1.0)
            >>> number_line.point_to_number([[0.5, 0, 0], [1, 0, 0], [1.5, 0, 0]])
            array([0.5, 1. , 1.5])

        """
        point = np.asarray(point)
        start, end = self.get_start_and_end()
        unit_vect = normalize(end - start)
        proportion = np.dot(point - start, unit_vect) / np.dot(end - start, unit_vect)
        return interpolate(self.x_min, self.x_max, proportion)

    def n2p(self, number: float | np.ndarray) -> np.ndarray:
        """Abbreviation for :meth:`~.NumberLine.number_to_point`."""
        return self.number_to_point(number)

    def p2n(self, point: Sequence[float]) -> float:
        """Abbreviation for :meth:`~.NumberLine.point_to_number`."""
        return self.point_to_number(point)

    def get_unit_size(self) -> float:
        return self.get_length() / (self.x_range[1] - self.x_range[0])

    def get_unit_vector(self) -> np.ndarray:
        return super().get_unit_vector() * self.unit_size

    def get_number_mobject(
        self,
        x: float,
        direction: Sequence[float] | None = None,
        buff: float | None = None,
        font_size: float | None = None,
        label_constructor: VMobject | None = None,
        **number_config,
    ) -> VMobject:
        """Generates a positioned :class:`~.DecimalNumber` mobject
        generated according to ``label_constructor``.

        Parameters
        ----------
        x
            The x-value at which the mobject should be positioned.
        direction
            Determines the direction at which the label is positioned next to the line.
        buff
            The distance of the label from the line.
        font_size
            The font size of the label mobject.
        label_constructor
            The :class:`~.VMobject` class that will be used to construct the label.
            Defaults to the ``label_constructor`` attribute of the number line
            if not specified.

        Returns
        -------
        :class:`~.DecimalNumber`
            The positioned mobject.
        """
        number_config = merge_dicts_recursively(
            self.decimal_number_config,
            number_config,
        )
        if direction is None:
            direction = self.label_direction
        if buff is None:
            buff = self.line_to_number_buff
        if font_size is None:
            font_size = self.font_size
        if label_constructor is None:
            label_constructor = self.label_constructor

        num_mob = DecimalNumber(
            x, font_size=font_size, mob_class=label_constructor, **number_config
        )

        num_mob.next_to(self.number_to_point(x), direction=direction, buff=buff)
        if x < 0 and self.label_direction[0] == 0:
            # Align without the minus sign
            num_mob.shift(num_mob[0].width * LEFT / 2)
        return num_mob

    def get_number_mobjects(self, *numbers, **kwargs) -> VGroup:
        if len(numbers) == 0:
            numbers = self.default_numbers_to_display()
        return VGroup([self.get_number_mobject(number, **kwargs) for number in numbers])

    def get_labels(self) -> VGroup:
        return self.get_number_mobjects()

    def add_numbers(
        self,
        x_values: Iterable[float] | None = None,
        excluding: Iterable[float] | None = None,
        font_size: float | None = None,
        label_constructor: VMobject | None = None,
        **kwargs,
    ):
        """Adds :class:`~.DecimalNumber` mobjects representing their position
        at each tick of the number line. The numbers can be accessed after creation
        via ``self.numbers``.

        Parameters
        ----------
        x_values
            An iterable of the values used to position and create the labels.
            Defaults to the output produced by :meth:`~.NumberLine.get_tick_range`
        excluding
            A list of values to exclude from :attr:`x_values`.
        font_size
            The font size of the labels. Defaults to the ``font_size`` attribute
            of the number line.
        label_constructor
            The :class:`~.VMobject` class that will be used to construct the label.
            Defaults to the ``label_constructor`` attribute of the number line
            if not specified.
        """
        if x_values is None:
            x_values = self.get_tick_range()

        if excluding is None:
            excluding = self.numbers_to_exclude

        if font_size is None:
            font_size = self.font_size

        if label_constructor is None:
            label_constructor = self.label_constructor

        numbers = VGroup()
        for x in x_values:
            if x in excluding:
                continue
            numbers.add(
                self.get_number_mobject(
                    x,
                    font_size=font_size,
                    label_constructor=label_constructor,
                    **kwargs,
                )
            )
        self.add(numbers)
        self.numbers = numbers
        return self

    def add_labels(
        self,
        dict_values: dict[float, str | float | VMobject],
        direction: Sequence[float] = None,
        buff: float | None = None,
        font_size: float | None = None,
        label_constructor: VMobject | None = None,
    ):
        """Adds specifically positioned labels to the :class:`~.NumberLine` using a ``dict``.
        The labels can be accessed after creation via ``self.labels``.

        Parameters
        ----------
        dict_values
            A dictionary consisting of the position along the number line and the mobject to be added:
            ``{1: Tex("Monday"), 3: Tex("Tuesday")}``. :attr:`label_constructor` will be used
            to construct the labels if the value is not a mobject (``str`` or ``float``).
        direction
            Determines the direction at which the label is positioned next to the line.
        buff
            The distance of the label from the line.
        font_size
            The font size of the mobject to be positioned.
        label_constructor
            The :class:`~.VMobject` class that will be used to construct the label.
            Defaults to the ``label_constructor`` attribute of the number line
            if not specified.

        Raises
        ------
        AttributeError
            If the label does not have a ``font_size`` attribute, an ``AttributeError`` is raised.
        """
        direction = self.label_direction if direction is None else direction
        buff = self.line_to_number_buff if buff is None else buff
        font_size = self.font_size if font_size is None else font_size
        if label_constructor is None:
            label_constructor = self.label_constructor

        labels = VGroup()
        for x, label in dict_values.items():
            # TODO: remove this check and ability to call
            # this method via CoordinateSystem.add_coordinates()
            # must be explicitly called
            if isinstance(label, str) and label_constructor is MathTex:
                label = Tex(label)
            else:
                label = self._create_label_tex(label, label_constructor)

            if hasattr(label, "font_size"):
                label.font_size = font_size
            else:
                raise AttributeError(f"{label} is not compatible with add_labels.")
            label.next_to(self.number_to_point(x), direction=direction, buff=buff)
            labels.add(label)

        self.labels = labels
        self.add(labels)
        return self

    def _create_label_tex(
        self,
        label_tex: str | float | VMobject,
        label_constructor: Callable | None = None,
        **kwargs,
    ) -> VMobject:
        """Checks if the label is a :class:`~.VMobject`, otherwise, creates a
        label by passing ``label_tex`` to ``label_constructor``.

        Parameters
        ----------
        label_tex
            The label for which a mobject should be created. If the label already
            is a mobject, no new mobject is created.
        label_constructor
            Optional. A class or function returning a mobject when
            passing ``label_tex`` as an argument. If ``None`` is passed
            (the default), the label constructor from the :attr:`.label_constructor`
            attribute is used.

        Returns
        -------
        :class:`~.VMobject`
            The label.
        """
        if label_constructor is None:
            label_constructor = self.label_constructor
        if isinstance(label_tex, (VMobject, OpenGLVMobject)):
            return label_tex
        else:
            return label_constructor(label_tex, **kwargs)

    @staticmethod
    def _decimal_places_from_step(step) -> int:
        step = str(step)
        if "." not in step:
            return 0
        return len(step.split(".")[-1])

    def __matmul__(self, other: float):
        return self.n2p(other)

    def __rmatmul__(self, other: Point3DLike | Mobject):
        if isinstance(other, Mobject):
            other = other.get_center()
        return self.p2n(other)


class UnitInterval(NumberLine):
    def __init__(
        self,
        unit_size=10,
        numbers_with_elongated_ticks=None,
        decimal_number_config=None,
        **kwargs,
    ):
        numbers_with_elongated_ticks = (
            [0, 1]
            if numbers_with_elongated_ticks is None
            else numbers_with_elongated_ticks
        )

        decimal_number_config = (
            {
                "num_decimal_places": 1,
            }
            if decimal_number_config is None
            else decimal_number_config
        )

        super().__init__(
            x_range=(0, 1, 0.1),
            unit_size=unit_size,
            numbers_with_elongated_ticks=numbers_with_elongated_ticks,
            decimal_number_config=decimal_number_config,
            **kwargs,
        )
