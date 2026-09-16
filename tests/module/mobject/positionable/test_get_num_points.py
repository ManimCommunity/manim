from manim.mobject.abstract.positionable import Positionable
from tests.module.mobject.positionable.utils import PositionableWithFamily


def test_no_points() -> None:
    p = Positionable()
    assert p.get_num_points() == 0


def test_single() -> None:
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    assert p.get_num_points() == 3


def test_family() -> None:
    p = PositionableWithFamily([Positionable(), Positionable(), Positionable()])
    assert p.get_num_points() == 0

    p = PositionableWithFamily(
        [Positionable(), Positionable(), Positionable()]
    ).set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    assert p.get_num_points() == 3

    p = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    )
    assert p.get_num_points() == 0
