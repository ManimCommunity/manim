# TODO


import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from manim.typing import Vector3DLike
from tests.module.mobject.positionable.utils import ANCHOR_POINTS, CUBE_VERTICES


def test_no_points_default() -> None:
    p = Positionable()
    np.testing.assert_allclose(p.get_anchor(), (0, 0, 0))


@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_no_points_anchor(anchor: Vector3DLike) -> None:
    p = Positionable()
    np.testing.assert_allclose(p.get_anchor(direction=anchor), (0, 0, 0))


def test_default() -> None:
    p = Positionable().set_points(CUBE_VERTICES)
    np.testing.assert_allclose(p.get_anchor(), (0, 0, 0))

    p = Positionable().set_points([(0, 1, 2), (3, 4, 5), (6, 7, 8)])
    np.testing.assert_allclose(p.get_anchor(), (3, 4, 5))


@pytest.mark.parametrize("anchor", ANCHOR_POINTS)
def test_anchor(anchor: Vector3DLike) -> None:
    p = Positionable().set_points(CUBE_VERTICES)
    np.testing.assert_allclose(p.get_anchor(anchor), anchor)
