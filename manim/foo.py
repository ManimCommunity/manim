from manim.my_module.bar import Bar


def foo_1() -> None:
    """Prints hello."""
    print("Hello")

def foo_2(x: int) -> float:
    """Squares a number.

    Parameters
    ----------
    x
        The number.

    Returns
    -------
        The squared number.
    """
    return x * x


class FooBar(Bar):
    """The FooBar class."""
    def foo_bar(self) -> None:
        pass
