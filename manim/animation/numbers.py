"""Animations for changing numbers."""

__all__ = ["ChangingDecimal", "ChangeDecimalToValue"]


import typing

from ..animation.animation import Animation
from ..mobject.numbers import DecimalNumber
from ..utils.bezier import interpolate


class ChangingDecimal(Animation):
    def __init__(
        self,
        decimal_mob: DecimalNumber,
        number_update_func: typing.Callable[[float], float],
        suspend_mobject_updating: typing.Optional[bool] = False,
        **kwargs,
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
        self.mobject.set_value(self.number_update_func(self.rate_func(alpha)))


class ChangeDecimalToValue(ChangingDecimal):
    def __init__(
        self, decimal_mob: DecimalNumber, target_number: int, **kwargs
    ) -> None:
        start_number = decimal_mob.number
        super().__init__(
            decimal_mob, lambda a: interpolate(start_number, target_number, a), **kwargs
        )
