"""Animations for changing numbers."""

from __future__ import annotations

__all__ = ["ChangingDecimal", "ChangeDecimalToValue"]


import typing

from typing_extensions import Any

from manim.mobject.text.numbers import DecimalNumber

from ..animation.animation import Animation
from ..utils.bezier import interpolate


class ChangingDecimal(Animation):
    """Animates a DecimalNumber mobject by interpolating the value over time
    based on alpha value.

    Parameters
    ----------
    decimal_mob
        The DecimalNumber mobject to animate.
    number_update_func
        A function that returns the value to display based on current alpha.
    suspend_mobject_updating
        Whether to suspend mobject automatic updates during animation.
    **kwargs
        Additional keyword arguments passed to :class:`~.Animation`.

    Examples
    --------
    .. manim:: ChangingDecimalDemo

        class ChangingDecimalDemo(Scene):
            def construct(self):
                decimal = DecimalNumber(0)
                self.add(decimal)
                self.play(ChangingDecimal(decimal, lambda a: a * 100))
    """
    def __init__(
        self,
        decimal_mob: DecimalNumber,
        number_update_func: typing.Callable[[float], float],
        suspend_mobject_updating: bool = False,
        **kwargs: Any,
    ) -> None:
        self.check_validity_of_input(decimal_mob)
        self.number_update_func = number_update_func
        super().__init__(
            decimal_mob, suspend_mobject_updating=suspend_mobject_updating, **kwargs
        )

    def check_validity_of_input(self, decimal_mob: DecimalNumber) -> None:
        if not isinstance(decimal_mob, DecimalNumber):
            raise TypeError("ChangingDecimal can only take in a DecimalNumber")

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.set_value(self.number_update_func(self.rate_func(alpha)))  # type: ignore[attr-defined]


class ChangeDecimalToValue(ChangingDecimal):
    """Animates a DecimalNumber mobject from its current value to a target value.

    Parameters
    ----------
    decimal_mob
        The DecimalNumber mobject to animate.
    target_number
        The value to which to animate.
    **kwargs
        Additional keyword arguments passed to :class:`~.Animation`.

    Examples
    --------
    .. manim:: ChangeDecimalToValueDemo

        class ChangeDecimalToValueDemo(Scene):
            def construct(self):
                decimal = DecimalNumber(0)
                self.add(decimal)
                self.play(ChangeDecimalToValue(decimal, 100))
    """
    def __init__(
        self, decimal_mob: DecimalNumber, target_number: int, **kwargs: Any
    ) -> None:
        start_number = decimal_mob.number
        super().__init__(
            decimal_mob, lambda a: interpolate(start_number, target_number, a), **kwargs
        )
