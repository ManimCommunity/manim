"""Mobjects representing numbers."""

from __future__ import annotations

__all__ = ["DecimalNumber", "Integer", "Variable"]

from typing import Sequence

import numpy as np

from manim import config
from manim.constants import *
from manim.mobject.opengl.opengl_compatibility import ConvertToOpenGL
from manim.mobject.text.tex_mobject import MathTex, SingleStringMathTex, Tex
from manim.mobject.text.text_mobject import Text
from manim.mobject.types.vectorized_mobject import VMobject
from manim.mobject.value_tracker import ValueTracker

string_to_mob_map = {}

__all__ = ["DecimalNumber", "Integer", "Variable"]


class DecimalNumber(VMobject, metaclass=ConvertToOpenGL):
    """An mobject representing a decimal number.

    Parameters
    ----------
    number
        The numeric value to be displayed. It can later be modified using :meth:`.set_value`.
    num_decimal_places
        The number of decimal places after the decimal separator. Values are automatically rounded.
    mob_class
        The class for rendering digits and units, by default :class:`.MathTex`.
    include_sign
        Set to ``True`` to include a sign for positive numbers and zero.
    group_with_commas
        When ``True`` thousands groups are separated by commas for readability.
    digit_buff_per_font_unit
        Additional spacing between digits. Scales with font size.
    show_ellipsis
        When a number has been truncated by rounding, indicate with an ellipsis (``...``).
    unit
        A unit string which can be placed to the right of the numerical values.
    unit_buff_per_font_unit
        An additional spacing between the numerical values and the unit. A value
        of ``unit_buff_per_font_unit=0.003`` gives a decent spacing. Scales with font size.
    include_background_rectangle
        Adds a background rectangle to increase contrast on busy scenes.
    edge_to_fix
        Assuring right- or left-alignment of the full object.
    font_size
        Size of the font.

    Examples
    --------

    .. manim:: MovingSquareWithUpdaters

        class MovingSquareWithUpdaters(Scene):
            def construct(self):
                decimal = DecimalNumber(
                    0,
                    show_ellipsis=True,
                    num_decimal_places=3,
                    include_sign=True,
                    unit=r"\text{M-Units}",
                    unit_buff_per_font_unit=0.003
                )
                square = Square().to_edge(UP)

                decimal.add_updater(lambda d: d.next_to(square, RIGHT))
                decimal.add_updater(lambda d: d.set_value(square.get_center()[1]))
                self.add(square, decimal)
                self.play(
                    square.animate.to_edge(DOWN),
                    rate_func=there_and_back,
                    run_time=5,
                )
                self.wait()

    """

    def __init__(
        self,
        number: float = 0,
        num_decimal_places: int = 2,
        mob_class: VMobject = MathTex,
        include_sign: bool = False,
        group_with_commas: bool = True,
        digit_buff_per_font_unit: float = 0.001,
        show_ellipsis: bool = False,
        unit: str | None = None,  # Aligned to bottom unless it starts with "^"
        unit_buff_per_font_unit: float = 0,
        include_background_rectangle: bool = False,
        edge_to_fix: Sequence[float] = LEFT,
        font_size: float = DEFAULT_FONT_SIZE,
        stroke_width: float = 0,
        fill_opacity: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs, stroke_width=stroke_width)
        self.number = number
        self.num_decimal_places = num_decimal_places
        self.include_sign = include_sign
        self.mob_class = mob_class
        self.group_with_commas = group_with_commas
        self.digit_buff_per_font_unit = digit_buff_per_font_unit
        self.show_ellipsis = show_ellipsis
        self.unit = unit
        self.unit_buff_per_font_unit = unit_buff_per_font_unit
        self.include_background_rectangle = include_background_rectangle
        self.edge_to_fix = edge_to_fix
        self._font_size = font_size
        self.fill_opacity = fill_opacity

        self.initial_config = kwargs.copy()
        self.initial_config.update(
            {
                "num_decimal_places": num_decimal_places,
                "include_sign": include_sign,
                "group_with_commas": group_with_commas,
                "digit_buff_per_font_unit": digit_buff_per_font_unit,
                "show_ellipsis": show_ellipsis,
                "unit": unit,
                "unit_buff_per_font_unit": unit_buff_per_font_unit,
                "include_background_rectangle": include_background_rectangle,
                "edge_to_fix": edge_to_fix,
                "font_size": font_size,
                "stroke_width": stroke_width,
                "fill_opacity": fill_opacity,
            },
        )

        self._set_submobjects_from_number(number)
        self.init_colors()

    @property
    def font_size(self):
        """The font size of the tex mobject."""
        return self.height / self.initial_height * self._font_size

    @font_size.setter
    def font_size(self, font_val):
        if font_val <= 0:
            raise ValueError("font_size must be greater than 0.")
        elif self.height > 0:
            # sometimes manim generates a SingleStringMathex mobject with 0 height.
            # can't be scaled regardless and will error without the elif.

            # scale to a factor of the initial height so that setting
            # font_size does not depend on current size.
            self.scale(font_val / self.font_size)

    def _set_submobjects_from_number(self, number):
        self.number = number
        self.submobjects = []

        num_string = self._get_num_string(number)
        self.add(*(map(self._string_to_mob, num_string)))

        # Add non-numerical bits
        if self.show_ellipsis:
            self.add(
                self._string_to_mob("\\dots", SingleStringMathTex, color=self.color),
            )

        self.arrange(
            buff=self.digit_buff_per_font_unit * self._font_size,
            aligned_edge=DOWN,
        )

        if self.unit is not None:
            self.unit_sign = self._string_to_mob(self.unit, SingleStringMathTex)
            self.add(
                self.unit_sign.next_to(
                    self,
                    direction=RIGHT,
                    buff=(self.unit_buff_per_font_unit + self.digit_buff_per_font_unit)
                    * self._font_size,
                    aligned_edge=DOWN,
                )
            )

        self.move_to(ORIGIN)

        # Handle alignment of parts that should be aligned
        # to the bottom
        for i, c in enumerate(num_string):
            if c == "-" and len(num_string) > i + 1:
                self[i].align_to(self[i + 1], UP)
                self[i].shift(self[i + 1].height * DOWN / 2)
            elif c == ",":
                self[i].shift(self[i].height * DOWN / 2)
        if self.unit and self.unit.startswith("^"):
            self.unit_sign.align_to(self, UP)

        # track the initial height to enable scaling via font_size
        self.initial_height = self.height

        if self.include_background_rectangle:
            self.add_background_rectangle()

    def _get_num_string(self, number):
        if isinstance(number, complex):
            formatter = self._get_complex_formatter()
        else:
            formatter = self._get_formatter()
        num_string = formatter.format(number)

        rounded_num = np.round(number, self.num_decimal_places)
        if num_string.startswith("-") and rounded_num == 0:
            if self.include_sign:
                num_string = "+" + num_string[1:]
            else:
                num_string = num_string[1:]

        return num_string

    def _string_to_mob(self, string: str, mob_class: VMobject | None = None, **kwargs):
        if mob_class is None:
            mob_class = self.mob_class

        if string not in string_to_mob_map:
            string_to_mob_map[string] = mob_class(string, **kwargs)
        mob = string_to_mob_map[string].copy()
        mob.font_size = self._font_size
        return mob

    def _get_formatter(self, **kwargs):
        """
        Configuration is based first off instance attributes,
        but overwritten by any kew word argument.  Relevant
        key words:
        - include_sign
        - group_with_commas
        - num_decimal_places
        - field_name (e.g. 0 or 0.real)
        """
        config = {
            attr: getattr(self, attr)
            for attr in [
                "include_sign",
                "group_with_commas",
                "num_decimal_places",
            ]
        }
        config.update(kwargs)
        return "".join(
            [
                "{",
                config.get("field_name", ""),
                ":",
                "+" if config["include_sign"] else "",
                "," if config["group_with_commas"] else "",
                ".",
                str(config["num_decimal_places"]),
                "f",
                "}",
            ],
        )

    def _get_complex_formatter(self):
        return "".join(
            [
                self._get_formatter(field_name="0.real"),
                self._get_formatter(field_name="0.imag", include_sign=True),
                "i",
            ],
        )

    def set_value(self, number: float):
        """Set the value of the :class:`~.DecimalNumber` to a new number.

        Parameters
        ----------
        number
            The value that will overwrite the current number of the :class:`~.DecimalNumber`.

        """
        # creates a new number mob via `set_submobjects_from_number`
        # then matches the properties (color, font_size, etc...)
        # of the previous mobject to the new one

        # old_family needed with cairo
        old_family = self.get_family()

        old_font_size = self.font_size
        move_to_point = self.get_edge_center(self.edge_to_fix)
        old_submobjects = self.submobjects

        self._set_submobjects_from_number(number)
        self.font_size = old_font_size
        self.move_to(move_to_point, self.edge_to_fix)
        for sm1, sm2 in zip(self.submobjects, old_submobjects):
            sm1.match_style(sm2)

        if config.renderer == RendererType.CAIRO:
            for mob in old_family:
                # Dumb hack...due to how scene handles families
                # of animated mobjects
                # for compatibility with updaters to not leave first number in place while updating,
                # not needed with opengl renderer
                mob.points[:] = 0

        self.init_colors()
        return self

    def get_value(self):
        return self.number

    def increment_value(self, delta_t=1):
        self.set_value(self.get_value() + delta_t)


class Integer(DecimalNumber):
    """A class for displaying Integers.

    Examples
    --------

    .. manim:: IntegerExample
        :save_last_frame:

        class IntegerExample(Scene):
            def construct(self):
                self.add(Integer(number=2.5).set_color(ORANGE).scale(2.5).set_x(-0.5).set_y(0.8))
                self.add(Integer(number=3.14159, show_ellipsis=True).set_x(3).set_y(3.3).scale(3.14159))
                self.add(Integer(number=42).set_x(2.5).set_y(-2.3).set_color_by_gradient(BLUE, TEAL).scale(1.7))
                self.add(Integer(number=6.28).set_x(-1.5).set_y(-2).set_color(YELLOW).scale(1.4))
    """

    def __init__(self, number=0, num_decimal_places=0, **kwargs):
        super().__init__(number=number, num_decimal_places=num_decimal_places, **kwargs)

    def get_value(self):
        return int(np.round(super().get_value()))


class Variable(VMobject, metaclass=ConvertToOpenGL):
    """A class for displaying text that shows "label = value" with
    the value continuously updated from a :class:`~.ValueTracker`.

    Parameters
    ----------
    var
        The initial value you need to keep track of and display.
    label
        The label for your variable. Raw strings are convertex to :class:`~.MathTex` objects.
    var_type
        The class used for displaying the number. Defaults to :class:`DecimalNumber`.
    num_decimal_places
        The number of decimal places to display in your variable. Defaults to 2.
        If `var_type` is an :class:`Integer`, this parameter is ignored.
    kwargs
            Other arguments to be passed to `~.Mobject`.

    Attributes
    ----------
    label : Union[:class:`str`, :class:`~.Tex`, :class:`~.MathTex`, :class:`~.Text`, :class:`~.SingleStringMathTex`]
        The label for your variable, for example ``x = ...``.
    tracker : :class:`~.ValueTracker`
        Useful in updating the value of your variable on-screen.
    value : Union[:class:`DecimalNumber`, :class:`Integer`]
        The tex for the value of your variable.

    Examples
    --------
    Normal usage::

        # DecimalNumber type
        var = 0.5
        on_screen_var = Variable(var, Text("var"), num_decimal_places=3)
        # Integer type
        int_var = 0
        on_screen_int_var = Variable(int_var, Text("int_var"), var_type=Integer)
        # Using math mode for the label
        on_screen_int_var = Variable(int_var, "{a}_{i}", var_type=Integer)

    .. manim:: VariablesWithValueTracker

        class VariablesWithValueTracker(Scene):
            def construct(self):
                var = 0.5
                on_screen_var = Variable(var, Text("var"), num_decimal_places=3)

                # You can also change the colours for the label and value
                on_screen_var.label.set_color(RED)
                on_screen_var.value.set_color(GREEN)

                self.play(Write(on_screen_var))
                # The above line will just display the variable with
                # its initial value on the screen. If you also wish to
                # update it, you can do so by accessing the `tracker` attribute
                self.wait()
                var_tracker = on_screen_var.tracker
                var = 10.5
                self.play(var_tracker.animate.set_value(var))
                self.wait()

                int_var = 0
                on_screen_int_var = Variable(
                    int_var, Text("int_var"), var_type=Integer
                ).next_to(on_screen_var, DOWN)
                on_screen_int_var.label.set_color(RED)
                on_screen_int_var.value.set_color(GREEN)

                self.play(Write(on_screen_int_var))
                self.wait()
                var_tracker = on_screen_int_var.tracker
                var = 10.5
                self.play(var_tracker.animate.set_value(var))
                self.wait()

                # If you wish to have a somewhat more complicated label for your
                # variable with subscripts, superscripts, etc. the default class
                # for the label is MathTex
                subscript_label_var = 10
                on_screen_subscript_var = Variable(subscript_label_var, "{a}_{i}").next_to(
                    on_screen_int_var, DOWN
                )
                self.play(Write(on_screen_subscript_var))
                self.wait()

    .. manim:: VariableExample

        class VariableExample(Scene):
            def construct(self):
                start = 2.0

                x_var = Variable(start, 'x', num_decimal_places=3)
                sqr_var = Variable(start**2, 'x^2', num_decimal_places=3)
                Group(x_var, sqr_var).arrange(DOWN)

                sqr_var.add_updater(lambda v: v.tracker.set_value(x_var.tracker.get_value()**2))

                self.add(x_var, sqr_var)
                self.play(x_var.tracker.animate.set_value(5), run_time=2, rate_func=linear)
                self.wait(0.1)

    """

    def __init__(
        self,
        var: float,
        label: str | Tex | MathTex | Text | SingleStringMathTex,
        var_type: DecimalNumber | Integer = DecimalNumber,
        num_decimal_places: int = 2,
        **kwargs,
    ):
        self.label = MathTex(label) if isinstance(label, str) else label
        equals = MathTex("=").next_to(self.label, RIGHT)
        self.label.add(equals)

        self.tracker = ValueTracker(var)

        if var_type == DecimalNumber:
            self.value = DecimalNumber(
                self.tracker.get_value(),
                num_decimal_places=num_decimal_places,
            )
        elif var_type == Integer:
            self.value = Integer(self.tracker.get_value())

        self.value.add_updater(lambda v: v.set_value(self.tracker.get_value())).next_to(
            self.label,
            RIGHT,
        )

        super().__init__(**kwargs)
        self.add(self.label, self.value)
