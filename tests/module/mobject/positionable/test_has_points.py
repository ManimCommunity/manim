"""Tests the `Positionable.has_points` method."""

from manim.mobject.abstract.positionable import Positionable
from tests.module.mobject.positionable.utils import PositionableWithFamily


def test_no_points() -> None:
    """Tests whether `has_points` returns `False` for an object with no points."""
    p = Positionable()
    assert not p.has_points()


def test_single() -> None:
    """Tests whether `has_points` returns `True` for a simple object with points."""
    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    assert p.has_points()


def test_family() -> None:
    """Tests whether `has_points` does not take family members into account."""
    p = PositionableWithFamily([Positionable(), Positionable(), Positionable()])
    assert not p.has_points()

    p = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    ).set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    assert p.has_points()

    p = PositionableWithFamily(
        [
            Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)]),
            Positionable().set_points([(9, 10, 11), (12, 13, 14), (15, 16, 17)]),
            Positionable().set_points([(18, 19, 20), (21, 22, 23), (24, 25, 26)]),
        ]
    )
    assert not p.has_points()
