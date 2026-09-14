from __future__ import annotations

import numpy as np
import pytest

from manim.utils.space_ops import *
from manim.utils.space_ops import shoelace


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
    np.testing.assert_equal(angle_of_vector(np.zeros(3)), 0.0)


def test_angle_of_vector_vectorized():
    rng = np.random.default_rng()
    vec = rng.standard_normal((4, 10))
    ref = [np.angle(complex(*v[:2])) for v in vec.T]
    np.testing.assert_array_equal(ref, angle_of_vector(vec))


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


def test_triangulation_ring_connection():
    verts = np.array(
        [
            # outer ring
            [-2, -2, 0],
            [2, -2, 0],
            [2, 2, 0],
            [-2, 2, 0],
            # inner ring (hole)
            [-0.5, -1.5, 0],
            [-0.5, -0.5, 0],
            [-1.5, -0.5, 0],
            [-1.5, -1.5, 0],
        ],
        dtype=float,
    )
    ring_ends = [4, 8]

    triangulation = earclip_triangulation(verts, ring_ends)

    assert len(triangulation) > 0
    assert len(triangulation) % 3 == 0
    assert min(triangulation) >= 0
    assert max(triangulation) < len(verts)


@pytest.mark.parametrize(
    ("vec", "expected"),
    [
        (np.array([1, 2, 3]), [0.26726124, 0.53452248, 0.80178373]),
        ((-1, -2, -3), [-0.26726124, -0.53452248, -0.80178373]),
        ((0.1, 0.1, 0), [0.70710678, 0.70710678, 0]),
        (np.array([10]), [1]),
        (
            np.array([0, 1, 2, 3, 4, 5]),
            [0, 0.13483997, 0.26967994, 0.40451992, 0.53935989, 0.67419986],
        ),
    ],
)
def test_normalize_nonzero_vector(vec, expected):
    normalized_vec = normalize(vec)
    assert np.allclose(normalized_vec, expected)

    # check that fallback vector is ignored when the input vector is non-zero
    fallback_vec = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    normalized_vec_with_fallback = normalize(vec, fall_back=fallback_vec)
    assert np.all(normalized_vec_with_fallback == normalized_vec)


@pytest.mark.parametrize(
    "vec",
    [
        np.array([0, 0, 0]),
        np.array([-0, -0, -0]),
        (0, 0, 0),
        np.array([0]),
        np.array([0, 0, 0, 0, 0]),
    ],
)
def test_normalize_zero_vector(vec):
    normalized_zero_vec = normalize(vec)
    assert np.allclose(normalized_zero_vec, np.zeros(len(vec)))

    # check that fallback vector is returned when the input vector is zero
    fallback_vec = np.array([1, 0, 0])
    normalized_zero_vec_with_fallback = normalize(vec, fall_back=fallback_vec)
    assert np.allclose(normalized_zero_vec_with_fallback, fallback_vec)
