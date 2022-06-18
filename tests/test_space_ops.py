from __future__ import annotations

import numpy as np
import pytest

from manim.utils.space_ops import *
from manim.utils.space_ops import shoelace, shoelace_direction


def test_rotate_vector():
    vec = np.array([0, 1, 0])
    rotated = rotate_vector(vec, np.pi / 2)
    assert np.round(rotated[0], 5) == -1.0
    assert not np.round(rotated[1:], 5).any()
    np.testing.assert_array_equal(rotate_vector(np.zeros(3), np.pi / 4), np.zeros(3))


def test_rotation_matrices():
    ang = np.pi / 6
    ax = np.array([1, 1, 1])
    np.testing.assert_array_equal(
        np.round(rotation_matrix(ang, ax, True), 5),
        np.round(
            np.array(
                [
                    [0.91068, -0.24402, 0.33333, 0.0],
                    [0.33333, 0.91068, -0.24402, 0.0],
                    [-0.24402, 0.33333, 0.91068, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            5,
        ),
    )
    np.testing.assert_array_equal(
        np.round(rotation_about_z(np.pi / 3), 5),
        np.array(
            [
                [0.5, -0.86603, 0.0],
                [0.86603, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        np.round(z_to_vector(np.array([1, 2, 3])), 5),
        np.array(
            [
                [0.96362, 0.0, 0.26726],
                [-0.14825, 0.83205, 0.53452],
                [-0.22237, -0.5547, 0.80178],
            ]
        ),
    )


def test_angle_of_vector():
    assert angle_of_vector(np.array([1, 1, 1])) == np.pi / 4
    assert (
        np.round(angle_between_vectors(np.array([1, 1, 1]), np.array([-1, 1, 1])), 5)
        == 1.23096
    )
    assert angle_of_vector(np.zeros(3)) == 0.0


def test_center_of_mass():
    np.testing.assert_array_equal(
        center_of_mass([[0, 0, 0], [1, 2, 3]]), np.array([0.5, 1.0, 1.5])
    )


def test_line_intersection():
    np.testing.assert_array_equal(
        line_intersection(
            [[0, 0, 0], [3, 3, 0]],
            [[0, 3, 0], [3, 0, 0]],
        ),
        np.array([1.5, 1.5, 0.0]),
    )
    with pytest.raises(ValueError):
        line_intersection(  # parallel lines
            [[0, 1, 0], [5, 1, 0]],
            [[0, 6, 0], [5, 6, 0]],
        )
    with pytest.raises(ValueError):
        line_intersection(  # lines not in xy-plane
            [[0, 0, 3], [3, 3, 3]],
            [[0, 3, 3], [3, 0, 3]],
        )
    with pytest.raises(ValueError):
        line_intersection(  # lines are equal
            [[2, 2, 0], [3, 1, 0]],
            [[2, 2, 0], [3, 1, 0]],
        )
    np.testing.assert_array_equal(
        line_intersection(  # lines with ends out of bounds
            [[0, 0, 0], [1, 1, 0]],
            [[0, 4, 0], [1, 3, 0]],
        ),
        np.array([2, 2, 0]),
    )


def test_shoelace():
    assert shoelace(np.array([[1, 2], [3, 4]])) == 6


def test_polar_coords():
    a = np.array([1, 1, 0])
    b = (2, np.pi / 2, np.pi / 2)
    np.testing.assert_array_equal(
        np.round(cartesian_to_spherical(a), 4),
        np.round([2**0.5, np.pi / 4, np.pi / 2], 4),
    )
    np.testing.assert_array_equal(
        np.round(spherical_to_cartesian(b), 4), np.array([0, 2, 0])
    )
