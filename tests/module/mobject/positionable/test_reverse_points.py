import numpy as np

from manim.mobject.abstract.positionable import Positionable
from tests.module.mobject.positionable.utils import PositionableWithFamily


def test_no_points() -> None:
    p = Positionable()
    p.reverse_points()
    np.testing.assert_allclose(p.points, np.zeros((0, 3)))


def test_single() -> None:
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.reverse_points()
    np.testing.assert_allclose(p.points, [(6, 7, 8), (3, 4, 5), (0, 1, 2)])


def test_family() -> None:
    p = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    ).set_points([(27, 28, 29), (30, 31, 32), (33, 34, 35)])
    p.reverse_points()

    np.testing.assert_allclose(p.points, [(33, 34, 35), (30, 31, 32), (27, 28, 29)])
    np.testing.assert_allclose(
        p.submobjects[0].points, [(6, 7, 8), (3, 4, 5), (0, 1, 2)]
    )
    np.testing.assert_allclose(
        p.submobjects[1].points, [(15, 16, 17), (12, 13, 14), (9, 10, 11)]
    )
    np.testing.assert_allclose(
        p.submobjects[2].points, [(24, 25, 26), (21, 22, 23), (18, 19, 20)]
    )
