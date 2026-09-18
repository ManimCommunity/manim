"""Tests the `Positionable.get_dim_size` method."""

import numpy as np
import pytest

from manim.mobject.abstract.positionable import Positionable
from tests.module.mobject.positionable.utils import CUBE_VERTICES, DIMENSIONS


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_no_points(dim: int) -> None:
    """Tests whether `get_dim_size` returns 0 for an object with no points"""
    p = Positionable()
    np.testing.assert_allclose(p.get_dim_size(dim), 0)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_line(dim: int) -> None:
    """Tests whether `get_dim_size` returns the correct size(s) for a simple object."""
    p = Positionable().set_points([(0, 0, 0), (1, 2, 3), (2, 4, 6)])
    np.testing.assert_allclose(p.get_dim_size(dim), (2, 4, 6)[dim])


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_cube(dim: int) -> None:
    """Tests whether `get_dim_size` returns the correct size(s) for a cube."""
    p = Positionable().set_points(CUBE_VERTICES)
    np.testing.assert_allclose(p.get_dim_size(dim), 2)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_triangle(dim: int) -> None:
    """Tests whether `get_dim_size` returns the correct size(s) for a triangle."""
    p = Positionable().set_points([(-1, 0, 0), (1, 0, 0), (0, 1, 0)])
    np.testing.assert_allclose(p.get_dim_size(dim), (2, 1, 0)[dim])
