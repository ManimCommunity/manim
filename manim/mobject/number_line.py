"""Mobjects representing number lines.


Examples
--------

.. manim:: NumberLineExamples
    :save_last_frame:

    class NumberLineExamples(Scene):
        def construct(self):
            txt1 = Text("Default Number Line", size=0.6).shift(3.5 * UP)
            num_line1 = NumberLine().next_to(txt1, direction=DOWN)
            self.add(txt1, num_line1)

            txt2 = Text("Number Line with a tip", size=0.6).next_to(
                num_line1, direction=DOWN, buff=2 * DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
            )
            num_line2 = NumberLine(include_tip=True).next_to(txt2, direction=DOWN)
            self.add(txt2, num_line2)

            txt3 = Text("Number Line with labels", size=0.6).next_to(
                num_line2, direction=DOWN, buff=2 * DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
            )
            num_line3 = NumberLine(include_numbers=True).next_to(txt3, direction=DOWN)
            self.add(txt3, num_line3)

            txt4 = Text("Number Line with specific labels", size=0.6).next_to(
                num_line3, direction=DOWN, buff=2 * DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
            )
            num_line4 = NumberLine(include_numbers=False).next_to(txt4, direction=DOWN)
            self.add(txt4, num_line4)

            x_numbers = num_line4.get_number_mobjects(1, 2, -2)
            x_numbers.set_color(ORANGE)
            self.add(x_numbers)

            txt5 = Text("Number Line with unit interval", size=0.6).next_to(
                num_line4, direction=DOWN, buff=2 * DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
            )
            num_line5 = UnitInterval().next_to(txt5, direction=DOWN)
            self.add(txt5, num_line5)
"""


__all__ = ["NumberLine", "UnitInterval"]


import operator as op

from .. import config
from ..constants import *
from ..mobject.geometry import Line
from ..mobject.numbers import DecimalNumber
from ..mobject.types.vectorized_mobject import VGroup
from ..utils.bezier import interpolate
from ..utils.config_ops import merge_dicts_recursively
from ..utils.simple_functions import fdiv
from ..utils.space_ops import normalize
from ..utils.color import Colors, LIGHT_GREY


class NumberLine(Line):
    """Creates a number line"""

    def __init__(
        self,
        color: Colors = LIGHT_GREY,
        unit_size: int = 1,
        width: float = None,
        include_ticks: bool = True,
        tick_size: float = 0.1,
        tick_frequency: int = 1,
        # Defaults to value near x_min s.t. 0 is a tick
        # TODO, rename this
        leftmost_tick: float = None,
        # Change name
        numbers_with_elongated_ticks: typing.List[float] = [0],
        include_numbers: bool = False,
        numbers_to_show: typing.List[float] = None,
        longer_tick_multiple: int = 2,
        number_at_center: float = 0,
        number_scale_val: float = 0.75,
        label_direction: np.ndarray = DOWN,
        line_to_number_buff: float = MED_SMALL_BUFF,
        include_tip: bool = False,
        tip_width: float = 0.25,
        tip_height: float = 0.25,
        add_start: float = 0,  # extend number line by this amount at its starting point
        add_end: float = 0,  # extend number line by this amount at its end point
        decimal_number_config: dict = {"num_decimal_places": 0},
        exclude_zero_from_default_numbers: bool = False,
        x_min: float = -config["frame_x_radius"],
        x_max: float = config["frame_x_radius"],
        **kwargs
    ):
        """

        Parameters
        ----------
        color : :class:`~.Colors`, optional
            the color for the number line, by default LIGHT_GREY
        unit_size : :class:`int`, optional
            the unit size, by default 1
        width : :class:`float`, optional
            the width of the number line, by default None
        include_ticks : :class:`bool`, optional
            True if should include the tick marks, by default True
        tick_size : :class:`float`, optional
            the size of the ticks, by default 0.1
        tick_frequency : :class:`int`, optional
            the frequency of the tick marks, by default 1
        leftmost_tick : :class:`float`, optional
            the leftmost tick, by default None
        numbers_with_elongated_ticks : :class:`list`, optional
            the numbers with elongated ticks, by default [0]
        include_numbers : :class:`bool`, optional
            True if should display numbers, by default False
        numbers_to_show : :class:`list`, optional
            the list of numbers to display, by default None
        longer_tick_multiple : :class:`int`, optional
            the scale factor for longer ticks, by default 2
        number_at_center : :class:`int`, optional
            the number at the center, by default 0
        number_scale_val : :class:`float`, optional
            the scale factor for the numbers, by default 0.75
        label_direction : :class:`np.ndarray`, optional
            the direction of the label, by default DOWN
        line_to_number_buff : :class:`float`, optional
            the buffer between the line and numbers, by default MED_SMALL_BUFF
        include_tip : :class:`bool`, optional
            True if should include tip, by default False
        tip_width : :class:`float`, optional
            the width of the tip, by default 0.25
        tip_height : :class:`float`, optional
            the height of the tip, by default 0.25
        exclude_zero_from_default_numbers : :class:`bool`, optional
            True if should exclude zero from numbers/labels , by default False
        x_min : :class:`float`, optional
            the min number, by default -config["frame_x_radius"]
        x_max : :class:`float`, optional
            the max number, by default config["frame_x_radius"]
        """
        self.unit_size = unit_size
        self.include_ticks = include_ticks
        self.tick_size = tick_size
        self.tick_frequency = tick_frequency
        self.leftmost_tick = leftmost_tick
        self.numbers_with_elongated_ticks = numbers_with_elongated_ticks
        self.include_numbers = include_numbers
        self.numbers_to_show = numbers_to_show
        self.longer_tick_multiple = longer_tick_multiple
        self.number_at_center = number_at_center
        self.number_scale_val = number_scale_val
        self.label_direction = label_direction
        self.line_to_number_buff = line_to_number_buff
        self.include_tip = include_tip
        self.tip_width = tip_width
        self.tip_height = tip_height
        self.add_start = add_start
        self.add_end = add_end
        self.decimal_number_config = decimal_number_config
        self.exclude_zero_from_default_numbers = exclude_zero_from_default_numbers
        self.x_min = x_min
        self.x_max = x_max

        start = unit_size * self.x_min * RIGHT
        end = unit_size * self.x_max * RIGHT
        Line.__init__(
            self,
            start=start - add_start * RIGHT,
            end=end + add_end * RIGHT,
            color=color,
            **kwargs,
        )
        if width is not None:
            self.width = width
            self.unit_size = self.get_unit_size()
        self.shift(-self.number_to_point(self.number_at_center))

        self.init_leftmost_tick()
        if self.include_tip:
            self.add_tip()
        if self.include_ticks:
            self.add_tick_marks()
        if self.include_numbers:
            self.add_numbers()

    def init_leftmost_tick(self):
        if self.leftmost_tick is None:
            self.leftmost_tick = op.mul(
                self.tick_frequency, np.ceil(self.x_min / self.tick_frequency)
            )

    def add_tick_marks(self):
        """Adds the tick marks on the NumberLine"""
        tick_size = self.tick_size
        self.tick_marks = VGroup(
            *[self.get_tick(x, tick_size) for x in self.get_tick_numbers()]
        )
        big_tick_size = tick_size * self.longer_tick_multiple
        self.big_tick_marks = VGroup(
            *[
                self.get_tick(x, big_tick_size)
                for x in self.numbers_with_elongated_ticks
            ]
        )
        self.add(
            self.tick_marks,
            self.big_tick_marks,
        )

    def get_tick(self, x: float, size: float = None) -> Line:
        """Returns the tick Mobject (:class:`~.Line`)

        Parameters
        ----------
        x : :class:`float`
            the number corresponding to the tick mark
        size : :class:`float`, optional
            the size of the tick mark, by default None

        Returns
        -------
        :class:`~.Line`
            the tick mark (:class:`~.Line`) Mobject
        """
        if size is None:
            size = self.tick_size
        result = Line(size * DOWN, size * UP)
        result.rotate(self.get_angle())
        result.move_to(self.number_to_point(x))
        result.match_style(self)
        return result

    def get_tick_marks(self) -> VGroup:
        """Returns the tick marks on the NumberLine

        Returns
        -------
        :class:`~.VGroup`
            the VGroup containing the tick marks
        """
        return VGroup(
            *self.tick_marks,
            *self.big_tick_marks,
        )

    def get_tick_numbers(self) -> np.ndarray:
        """Returns the numbers corresponding to the tick marks

        Returns
        -------
        :class:`np.array`
            the numpy array containing the numbers
        """
        u = -1 if self.include_tip and self.add_end == 0 else 1
        return np.arange(
            self.leftmost_tick,
            self.x_max + u * self.tick_frequency / 2,
            self.tick_frequency,
        )

    def number_to_point(self, number: float) -> np.ndarray:
        """Converts the point on the screen to the number of NumberLine

        Parameters
        ----------
        number : :class:`float`
            the number to convert

        Returns
        -------
        :class:`np.ndarray`
            the point on the screen
        """
        alpha = float(number - self.x_min) / (self.x_max - self.x_min)
        return interpolate(
            self.get_start() + self.add_start * RIGHT,
            self.get_end() - self.add_end * RIGHT,
            alpha,
        )

    def point_to_number(self, point: np.ndarray) -> float:
        """Converts the point on the screen to the number on NumberLine

        Parameters
        ----------
        point : :class:`np.ndarray`
            the point to convert

        Returns
        -------
        :class:`float`
            the number on the NumberLine
        """
        start_point, end_point = self.get_start_and_end()
        full_vect = end_point - start_point
        unit_vect = normalize(full_vect)

        def distance_from_start(p):
            return np.dot(p - start_point, unit_vect)

        proportion = fdiv(distance_from_start(point), distance_from_start(end_point))
        return interpolate(self.x_min, self.x_max, proportion)

    def n2p(self, number: float) -> np.ndarray:
        """An alias for :meth:`~.NumberLine.number_to_point`"""
        return self.number_to_point(number)

    def p2n(self, point: np.ndarray) -> float:
        """An alias for :meth:`~.NumberLine.point_to_number`"""
        return self.point_to_number(point)

    def get_unit_size(self) -> float:
        """Returns the unit size of the NumberLine

        Returns
        -------
        :class:`float`
            the unit size
        """
        return self.get_length() / (self.x_max - self.x_min)

    def get_unit_vector(self) -> np.ndarray:
        """Returns a unit vector

        Returns
        -------
        :class:`np.ndarray`
            the unit vector scaled with the unit size of the NumberLine
        """
        return super().get_unit_vector() * self.unit_size

    def default_numbers_to_display(self) -> np.ndarray:
        """Returns the default numbers to display

        Returns
        -------
        :class:`np.array`
            the default numbers as numpy array
        """
        if self.numbers_to_show is not None:
            return self.numbers_to_show
        numbers = np.arange(
            np.floor(self.leftmost_tick),
            np.ceil(self.x_max),
        )
        if self.exclude_zero_from_default_numbers:
            numbers = numbers[numbers != 0]
        return numbers

    def get_number_mobject(
        self,
        number: float,
        number_config: dict = None,
        scale_val: float = None,
        direction: np.ndarray = None,
        buff: float = None,
    ) -> DecimalNumber:
        """Returns the :class:`~.DecimalNumber` mobject for the passed number

        Parameters
        ----------
        number : :class:`float`
            the number on the line
        number_config : :class:`dict`, optional
            the config to be passed to :class:`~.DecimalNumber`, by default None
        scale_val : :class:`float`, optional
            the scale factor for :class:`~.DecimalNumber` mobject, by default None
        direction : :class:`np.ndarray, optional
            the direction for the :class:`~.DecimalNumber` mobject, by default None
        buff : :class:`float`, optional
            the buffer between the number and the number line, by default None

        Returns
        -------
        :class:`~.DecimalNumber`
            The :class:`~.DecimalNumber` mobject for the passed number
        """
        number_config = merge_dicts_recursively(
            self.decimal_number_config,
            number_config or {},
        )
        if scale_val is None:
            scale_val = self.number_scale_val
        if direction is None:
            direction = self.label_direction
        buff = buff or self.line_to_number_buff
        num_mob = DecimalNumber(number, **number_config)
        num_mob.scale(scale_val)
        num_mob.next_to(self.number_to_point(number), direction=direction, buff=buff)
        return num_mob

    def get_number_mobjects(self, *numbers: float, **kwargs) -> VGroup:
        """Returns the labels (type :class:`~.DecimalNumber`) of the number line as a VGroup.

        Parameters
        ----------
        numbers : :class:`list[float]`
            the numbers to put

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing the :class:`~.DecimalNumber` mobjects.
        """
        if len(numbers) == 0:
            numbers = self.default_numbers_to_display()
        return VGroup(
            *[self.get_number_mobject(number, **kwargs) for number in numbers]
        )

    def get_labels(self) -> VGroup:
        """Returns the default labels (type :class:`~.DecimalNumber`) of the number line as a VGroup.

        Returns
        --------
        :class:`~.VGroup`
            VGroup containing the :class:`~.DecimalNumber` mobjects.
        """
        return self.get_number_mobjects()

    def add_numbers(self, *numbers: float, **kwargs) -> "NumberLine":
        """Add the numbers (labels) to the NumberLine

        Parameters
        ----------
        numbers : :class:`list[float]`
            the numbers to put

        Returns
        -------
        :class:`NumberLine`
            The current number line object (self).
        """
        self.numbers = self.get_number_mobjects(*numbers, **kwargs)
        self.add(self.numbers)
        return self


class UnitInterval(NumberLine):
    """Creates a NumberLine with a unit interval """

    def __init__(
        self,
        unit_size: int = 6,
        tick_frequency: float = 0.1,
        numbers_with_elongated_ticks: typing.List[float] = [0, 1],
        number_at_center: float = 0.5,
        decimal_number_config: dict = {
            "num_decimal_places": 1,
        },
        **kwargs
    ):
        """

        Parameters
        ----------
        unit_size : :class:`int`, optional
            the unit size, by default 6
        tick_frequency : :class:`float`, optional
            the tick frequency, by default 0.1
        numbers_with_elongated_ticks : :class:`list`, optional
            the numbers on elongated ticks, by default [0, 1]
        number_at_center : :class:`float`, optional
            the number to be put at the center, by default 0.5
        decimal_number_config : :class:`dict`, optional
            the configuration for :class:`DecimalNnumber`, by default { "num_decimal_places": 1, }
        """
        NumberLine.__init__(
            self,
            unit_size=unit_size,
            tick_frequency=tick_frequency,
            numbers_with_elongated_ticks=numbers_with_elongated_ticks,
            number_at_center=number_at_center,
            decimal_number_config=decimal_number_config,
            x_min=0,
            x_max=1,
            **kwargs,
        )
