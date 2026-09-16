import numpy as np

from manim.mobject.abstract.positionable import Positionable
from tests.module.mobject.positionable.utils import PositionableWithFamily


def test_single_to_single() -> None:
    p = Positionable()
    other = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    p.match_points(other)
    np.testing.assert_allclose(p.points, [(0, 1, 2), (3, 4, 5), (6, 7, 8)])


def test_single_to_family() -> None:
    p = Positionable()
    other = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    ).set_points([(27, 28, 29), (30, 31, 32), (33, 34, 35)])

    p.match_points(other)
    np.testing.assert_allclose(p.points, [(27, 28, 29), (30, 31, 32), (33, 34, 35)])


def test_family_to_single() -> None:
    p = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    ).set_points([(27, 28, 29), (30, 31, 32), (33, 34, 35)])
    other = Positionable().set_points([(36, 37, 38)])

    p.match_points(other)
    np.testing.assert_allclose(p.points, [(36, 37, 38)])
    np.testing.assert_allclose(
        p.submobjects[0].points, [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    )
    np.testing.assert_allclose(
        p.submobjects[1].points, [(9, 10, 11), (12, 13, 14), (15, 16, 17)]
    )
    np.testing.assert_allclose(
        p.submobjects[2].points, [(18, 19, 20), (21, 22, 23), (24, 25, 26)]
    )


def test_family_to_family() -> None:
    p = PositionableWithFamily(
        [
            Positionable().set_points([(1, 2, 3)]),
            Positionable().set_points([(4, 5, 6)]),
            Positionable().set_points([(7, 8, 9)]),
        ]
    ).set_points([(10, 11, 12)])
    other = PositionableWithFamily(
        [
            Positionable().set_points([(13, 14, 15)]),
            Positionable().set_points([(16, 17, 18)]),
            Positionable().set_points([(19, 20, 21)]),
        ]
    ).set_points([(22, 23, 24)])

    p.match_points(other)
    np.testing.assert_allclose(p.points, [(22, 23, 24)])
    np.testing.assert_allclose(p.submobjects[0].points, [(13, 14, 15)])
    np.testing.assert_allclose(p.submobjects[1].points, [(16, 17, 18)])
    np.testing.assert_allclose(p.submobjects[2].points, [(19, 20, 21)])
