"""Mobject representing a number line."""


__all__ = ["NumberLine", "UnitInterval"]


import numpy as np
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
from ..utils.color import LIGHT_GREY


class NumberLine(Line):
    def __init__(
        self,
        x_range=None,  # must be first
        length=None,
        unit_size=1,
        # ticks
        include_ticks=True,
        tick_size=0.1,
        numbers_with_elongated_ticks=[],
        longer_tick_multiple=2,
        # visuals
        color=LIGHT_GREY,
        rotation=0,
        stroke_width=2.0,
        # tip
        include_tip=False,
        tip_width=0.25,
        tip_height=0.25,
        # numbers
        include_numbers=False,
        label_direction=DOWN,
        line_to_number_buff=MED_SMALL_BUFF,
        decimal_number_config={"num_decimal_places": 0, "font_size": 24},
        numbers_to_exclude=[],
        # temp, because DecimalNumber() needs to be updated
        number_scale_value=0.75,
        **kwargs
    ):
        if x_range is None:
            x_range = [-config["frame_x_radius"], config["frame_x_radius"], 1.0]
        if len(x_range) == 2:
            # adds x_step if not specified. not sure how to feel about this. a user can't know default without peeking at source code
            x_range = [*x_range, 1]

        self.x_min, self.x_max, self.x_step = x_range
        self.length = length
        self.unit_size = unit_size
        # ticks
        self.include_ticks = include_ticks
        self.tick_size = tick_size
        self.numbers_with_elongated_ticks = numbers_with_elongated_ticks
        self.longer_tick_multiple = longer_tick_multiple
        # visuals
        self.stroke_width = stroke_width
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
        self.number_scale_value = number_scale_value

        super().__init__(
            self.x_min * RIGHT,
            self.x_max * RIGHT,
            stroke_width=self.stroke_width,
            color=self.color,
            **kwargs,
        )
        if self.length:
            self.rescale_to_fit(self.length, 0, stretch=False)
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
        if self.include_numbers:
            self.add_numbers(excluding=self.numbers_to_exclude)

    def rotate_about_zero(self, angle, axis=OUT, **kwargs):
        return self.rotate_about_number(0, angle, axis, **kwargs)

    def rotate_about_number(self, number, angle, axis=OUT, **kwargs):
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

    def get_tick(self, x, size=None):
        if size is None:
            size = self.tick_size
        result = Line(size * DOWN, size * UP)
        result.rotate(self.get_angle())
        result.move_to(self.number_to_point(x))
        result.match_style(self)
        return result

    def get_tick_marks(self):
        return VGroup(self.ticks)

    def get_tick_range(self):
        if self.include_tip:
            x_max = self.x_max
        else:
            x_max = self.x_max + 1e-6
        return np.arange(self.x_min, x_max, self.x_step)

    def number_to_point(self, number):
        alpha = float(number - self.x_min) / (self.x_max - self.x_min)
        return interpolate(self.get_start(), self.get_end(), alpha)

    def point_to_number(self, point):
        start, end = self.get_start_and_end()
        unit_vect = normalize(end - start)
        proportion = fdiv(
            np.dot(point - start, unit_vect),
            np.dot(end - start, unit_vect),
        )
        return interpolate(self.x_min, self.x_max, proportion)

    def n2p(self, number):
        """Abbreviation for number_to_point"""
        return self.number_to_point(number)

    def p2n(self, point):
        """Abbreviation for point_to_number"""
        return self.point_to_number(point)

    def get_unit_size(self):
        return self.get_length() / (self.x_max - self.x_min)

    def get_unit_vector(self):
        return super().get_unit_vector() * self.unit_size

    def get_number_mobject(self, x, direction=None, buff=None, **number_config):
        number_config = merge_dicts_recursively(
            self.decimal_number_config, number_config
        )
        if direction is None:
            direction = self.label_direction
        if buff is None:
            buff = self.line_to_number_buff

        num_mob = DecimalNumber(x, **number_config)
        # font_size does not exist yet in decimal number
        num_mob.scale(self.number_scale_value)

        num_mob.next_to(self.number_to_point(x), direction=direction, buff=buff)
        if x < 0 and self.label_direction[0] == 0:
            # Align without the minus sign
            num_mob.shift(num_mob[0].get_width() * LEFT / 2)
        return num_mob

    def get_number_mobjects(self, *numbers, **kwargs):
        if len(numbers) == 0:
            numbers = self.default_numbers_to_display()
        return VGroup([self.get_number_mobject(number, **kwargs) for number in numbers])

    def get_labels(self):
        return self.get_number_mobjects()

    def add_numbers(self, x_values=None, excluding=None, font_size=24, **kwargs):
        if x_values is None:
            x_values = self.get_tick_range()

        kwargs["font_size"] = font_size

        numbers = VGroup()
        for x in x_values:
            if x in self.numbers_to_exclude:
                continue
            if excluding is not None and x in excluding:
                continue
            numbers.add(self.get_number_mobject(x, **kwargs))
        self.add(numbers)
        self.numbers = numbers
        return numbers


class UnitInterval(NumberLine):
    def __init__(
        self,
        unit_size=10,
        tick_frequency=0.1,
        numbers_with_elongated_ticks=None,
        decimal_number_config=None,
        **kwargs
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

        NumberLine.__init__(
            self,
            x_range=[0, 1, 0.1],
            unit_size=unit_size,
            numbers_with_elongated_ticks=numbers_with_elongated_ticks,
            decimal_number_config=decimal_number_config,
            **kwargs,
        )
