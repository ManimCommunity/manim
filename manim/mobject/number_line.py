"""Mobject representing a number line."""

__all__ = ["NumberLine", "UnitInterval"]

from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Union

import numpy as np
from colour import Color

from manim.mobject.svg.tex_mobject import MathTex, Tex

from .. import config
from ..constants import *
from ..mobject.geometry import Line
from ..mobject.numbers import DecimalNumber
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.bezier import interpolate
from ..utils.color import LIGHT_GREY
from ..utils.config_ops import merge_dicts_recursively
from ..utils.deprecation import deprecated_params
from ..utils.simple_functions import fdiv
from ..utils.space_ops import normalize

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject


class NumberLine(Line):
    """Creates a number line with tick marks. Number ranges that include both negative and
    positive values will be generated from the 0 point, and may not include a tick at the min / max
    values as the tick locations are dependent on the step size.

    Parameters
    ----------
    x_range
        The :code:`[x_min, x_max, x_step]` values to create the line.
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
    color
        The color of the line.
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
    include_numbers
        Whether to add numbers to the tick marks. The number of decimal places is determined
        by the step size, this default can be overridden by ``decimal_number_config``.
    font size
        The size of the label mobjects. Defaults to 36.
    label_direction
        The specific position to which label mobjects are added on the line.
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
                [num6] = [num for num in l1.numbers if num.number == 6]
                num6.set_color(RED)
                l1.add(num6)

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

    Returns
    -------
    NumberLine
        The constructed number line.
    """

    @deprecated_params(
        params="number_scale_value",
        since="v0.10.0",
        until="v0.11.0",
        message="Use font_size instead.  To convert old scale factors to font size, multiply by 48.",
    )
    def __init__(
        self,
        x_range: Optional[Sequence[float]] = None,  # must be first
        length: Optional[float] = None,
        unit_size: float = 1,
        # ticks
        include_ticks: bool = True,
        tick_size: float = 0.1,
        numbers_with_elongated_ticks: Optional[Iterable[float]] = None,
        longer_tick_multiple: int = 2,
        exclude_origin_tick: bool = False,
        # visuals
        color: Color = LIGHT_GREY,
        rotation: float = 0,
        stroke_width: float = 2.0,
        # tip
        include_tip: bool = False,
        tip_width: float = 0.25,
        tip_height: float = 0.25,
        # numbers
        include_numbers: bool = False,
        font_size: float = 36,
        label_direction: Sequence[float] = DOWN,
        line_to_number_buff: float = MED_SMALL_BUFF,
        decimal_number_config: Optional[Dict] = None,
        numbers_to_exclude: Optional[Iterable[float]] = None,
        numbers_to_include: Optional[Iterable[float]] = None,
        **kwargs,
    ):
        # deprecation
        number_scale_value = kwargs.pop("number_scale_value", None)
        if number_scale_value is not None:
            self.font_size = number_scale_value * DEFAULT_FONT_SIZE * 0.75
        else:
            self.font_size = font_size

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

        self.x_min, self.x_max, self.x_step = x_range
        if decimal_number_config is None:
            decimal_number_config = {
                "num_decimal_places": self.decimal_places_from_step(),
            }

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
        self.color = color
        # tip
        self.include_tip = include_tip
        self.tip_width = tip_width
        self.tip_height = tip_height
        # numbers
        self.include_numbers = include_numbers
        self.label_direction = label_direction
        self.line_to_number_buff = line_to_number_buff
        self.decimal_number_config = decimal_number_config
        self.numbers_to_exclude = numbers_to_exclude
        self.numbers_to_include = numbers_to_include

        super().__init__(
            self.x_min * RIGHT,
            self.x_max * RIGHT,
            stroke_width=stroke_width,
            color=self.color,
            **kwargs,
        )
        if self.length:
            self.set_length(self.length)
            self.unit_size = self.get_unit_size()
        else:
            self.scale(self.unit_size)

        self.center()

        if self.include_tip:
            self.add_tip()
            self.tip.set_stroke(self.stroke_color, self.stroke_width)

        if self.include_ticks:
            self.add_ticks()

        self.rotate(self.rotation)
        if self.include_numbers or self.numbers_to_include is not None:
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
        ticks = VGroup()
        elongated_tick_size = self.tick_size * self.longer_tick_multiple
        for x in self.get_tick_range():
            size = self.tick_size
            if x in self.numbers_with_elongated_ticks:
                size = elongated_tick_size
            ticks.add(self.get_tick(x, size))
        self.add(ticks)
        self.ticks = ticks

    def get_tick(self, x: float, size: Optional[float] = None) -> Line:
        if size is None:
            size = self.tick_size
        result = Line(size * DOWN, size * UP)
        result.rotate(self.get_angle())
        result.move_to(self.number_to_point(x))
        result.match_style(self)
        return result

    def get_tick_marks(self) -> VGroup:
        return VGroup(self.ticks)

    def get_tick_range(self) -> np.ndarray:
        if self.include_tip:
            x_max = self.x_max
        else:
            x_max = self.x_max + 1e-6

        # Handle cases where min and max are both positive or both negative
        if self.x_min < x_max < 0 or self.x_max > self.x_min > 0:
            return np.arange(self.x_min, x_max, self.x_step)

        start_point = 0
        if self.exclude_origin_tick:
            start_point += self.x_step

        x_min_segment = (
            np.arange(start_point, np.abs(self.x_min) + 1e-6, self.x_step) * -1
        )
        x_max_segment = np.arange(start_point, x_max, self.x_step)

        return np.unique(np.concatenate((x_min_segment, x_max_segment)))

    def number_to_point(self, number: float) -> np.ndarray:
        alpha = float(number - self.x_min) / (self.x_max - self.x_min)
        return interpolate(self.get_start(), self.get_end(), alpha)

    def point_to_number(self, point: Sequence[float]) -> float:
        start, end = self.get_start_and_end()
        unit_vect = normalize(end - start)
        proportion = fdiv(
            np.dot(point - start, unit_vect),
            np.dot(end - start, unit_vect),
        )
        return interpolate(self.x_min, self.x_max, proportion)

    def n2p(self, number: float) -> np.ndarray:
        """Abbreviation for number_to_point"""
        return self.number_to_point(number)

    def p2n(self, point: Sequence[float]) -> float:
        """Abbreviation for point_to_number"""
        return self.point_to_number(point)

    def get_unit_size(self) -> float:
        return self.get_length() / (self.x_max - self.x_min)

    def get_unit_vector(self) -> np.ndarray:
        return super().get_unit_vector() * self.unit_size

    def get_number_mobject(
        self,
        x: float,
        direction: Optional[Sequence[float]] = None,
        buff: Optional[float] = None,
        font_size: Optional[float] = None,
        **number_config,
    ) -> DecimalNumber:
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

        num_mob = DecimalNumber(x, font_size=font_size, **number_config)

        num_mob.next_to(self.number_to_point(x), direction=direction, buff=buff)
        if x < 0 and self.label_direction[0] == 0:
            # Align without the minus sign
            num_mob.shift(num_mob[0].get_width() * LEFT / 2)
        return num_mob

    def get_number_mobjects(self, *numbers, **kwargs) -> VGroup:
        if len(numbers) == 0:
            numbers = self.default_numbers_to_display()
        return VGroup([self.get_number_mobject(number, **kwargs) for number in numbers])

    def get_labels(self) -> VGroup:
        return self.get_number_mobjects()

    def add_numbers(
        self,
        x_values: Optional[Iterable[float]] = None,
        excluding: Optional[Iterable[float]] = None,
        font_size: Optional[float] = None,
        **kwargs,
    ) -> VGroup:
        if x_values is None:
            x_values = self.get_tick_range()

        if excluding is None:
            excluding = self.numbers_to_exclude

        if font_size is None:
            font_size = self.font_size
        kwargs["font_size"] = font_size

        numbers = VGroup()
        for x in x_values:
            if x in excluding:
                continue
            numbers.add(self.get_number_mobject(x, **kwargs))
        self.add(numbers)
        self.numbers = numbers
        return self

    def add_labels(
        self,
        dict_values: Dict[float, Union[str, float, "Mobject"]],
        direction=None,
        buff=None,
        font_size=None,
    ):
        """Adds specifically positioned labels to the :class:`~.NumberLine` using a ``dict``."""
        direction = self.label_direction if direction is None else direction
        buff = self.line_to_number_buff if buff is None else buff
        font_size = self.font_size if font_size is None else font_size

        labels = VGroup()
        for x, label in dict_values.items():
            label = self.create_label_tex(label)
            if hasattr(label, "font_size"):
                label.font_size = font_size
            else:
                raise AttributeError(f"{label} is not compatible with add_labels.")
            label.next_to(self.number_to_point(x), direction=direction, buff=buff)
            labels.add(label)

        self.labels = labels
        self.add(labels)
        return self

    @staticmethod
    def create_label_tex(label_tex) -> "Mobject":
        """Checks if the label is a ``float``, ``int`` or a ``str`` and creates a :class:`~.MathTex`/:class:`~.Tex` label accordingly.

        Parameters
        ----------
        label_tex : The label to be compared against the above types.

        Returns
        -------
        :class:`~.Mobject`
            The label.
        """

        if isinstance(label_tex, (float, int)):
            label_tex = MathTex(label_tex)
        elif isinstance(label_tex, str):
            label_tex = Tex(label_tex)
        return label_tex

    def decimal_places_from_step(self) -> int:
        step_as_str = str(self.x_step)
        if "." not in step_as_str:
            return 0
        return len(step_as_str.split(".")[-1])


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
