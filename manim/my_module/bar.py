from typing import Literal


def bar_1() -> None:
    """Prints bar."""
    print("Bar")

def bar_2(x: float) -> int:
    """Squares a number.

    Parameters
    ----------
    x
        The number.

    Returns
    -------
        The squared number.
    """
    return int(x * x)


class Bar:
    """The bar class."""
    def bar(self, i: Literal[1, 2]) -> None | int:
        if i == 1:
            return bar_1()
        elif i == 2:
            return bar_2()
        else:
            raise ValueError()
