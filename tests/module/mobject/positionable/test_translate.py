import numpy as np

from manim.mobject.abstract.positionable import Positionable
from tests.module.mobject.positionable.utils import PositionableWithFamily


def test_no_points() -> None:
    p = Positionable()
    p.translate((1, 2, 3))
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


def test_simple() -> None:
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.translate((9, 10, 11))
    np.testing.assert_allclose(
        p.points,
        [
            (0 + 9, 1 + 10, 2 + 11),
            (3 + 9, 4 + 10, 5 + 11),
            (6 + 9, 7 + 10, 8 + 11),
        ],
    )


def test_family() -> None:
    p = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    ).set_points([(27, 28, 29), (30, 31, 32), (33, 34, 35)])
    p.translate((27, 28, 29))
