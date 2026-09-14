"""Animations for changing numbers."""

from __future__ import annotations

__all__ = ["ChangingDecimal", "ChangeDecimalToValue"]


from collections.abc import Callable
from typing import Any

from manim.mobject.text.numbers import DecimalNumber

from ..animation.animation import Animation
from ..utils.bezier import interpolate


class ChangingDecimal(Animation):
    """Animate a :class:`~.DecimalNumber` to values specified by a user-supplied function.

    Parameters
    ----------
    decimal_mob
        The :class:`~.DecimalNumber` instance to animate.
    number_update_func
        A function that returns the number to display at each point in the animation.
    suspend_mobject_updating
        If ``True``, the mobject is not updated outside this animation.

    Raises
    ------
    TypeError
        If ``decimal_mob`` is not an instance of :class:`~.DecimalNumber`.

    Examples
    --------

    .. manim:: ChangingDecimalExample

        class ChangingDecimalExample(Scene):
            def construct(self):
                number = DecimalNumber(0)
                self.add(number)
                self.play(
                    ChangingDecimal(
                        number,
                        lambda a: 5 * a,
                        run_time=3
                    )
                )
                self.wait()
    """

    def __init__(
        self,
        decimal_mob: DecimalNumber,
        number_update_func: Callable[[float], float],
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
    """Animate a :class:`~.DecimalNumber` to a target value using linear interpolation.

    Parameters
    ----------
    decimal_mob
        The :class:`~.DecimalNumber` instance to animate.
    target_number
        The target value to transition to.

    Examples
    --------

    .. manim:: ChangeDecimalToValueExample

        class ChangeDecimalToValueExample(Scene):
            def construct(self):
                number = DecimalNumber(0)
                self.add(number)
                self.play(ChangeDecimalToValue(number, 10, run_time=3))
                self.wait()
    """

    def __init__(
        self, decimal_mob: DecimalNumber, target_number: int, **kwargs: Any
    ) -> None:
        start_number = decimal_mob.number
        super().__init__(
            decimal_mob, lambda a: interpolate(start_number, target_number, a), **kwargs
        )
