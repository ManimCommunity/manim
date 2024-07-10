from __future__ import annotations

import numpy as np
import numpy.testing as nt
from _split_matrices import SPLIT_MATRICES
from _subdivision_matrices import SUBDIVISION_MATRICES

from manim.typing import ManimFloat
from manim.utils.bezier import (
    _get_subdivision_matrix,
    get_smooth_cubic_bezier_handle_points,
    partial_bezier_points,
    split_bezier,
    subdivide_bezier,
)

QUARTIC_BEZIER = np.array(
    [
        [-1, -1, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, -1, 0],
    ],
    dtype=float,
)


def test_partial_bezier_points() -> None:
    """Test that :func:`partial_bezierpoints`, both in the
    portion-matrix-building algorithm (degrees up to 3) and the
    fallback algorithm (degree 4), works correctly.
    """
    for degree, degree_dict in SUBDIVISION_MATRICES.items():
        n_points = degree + 1
        points = QUARTIC_BEZIER[:n_points]
        for n_divisions, subdivision_matrix in degree_dict.items():
            for i in range(n_divisions):
                a = i / n_divisions
                b = (i + 1) / n_divisions
                portion_matrix = subdivision_matrix[n_points * i : n_points * (i + 1)]
                nt.assert_allclose(
                    partial_bezier_points(points, a, b),
                    portion_matrix @ points,
                    atol=1e-15,  # Needed because of floating-point errors
                )


def test_split_bezier() -> None:
    """Test that :func:`split_bezier`, both in the
    split-matrix-building algorithm (degrees up to 3) and the
    fallback algorithm (degree 4), works correctly.
    """
    for degree, degree_dict in SPLIT_MATRICES.items():
        n_points = degree + 1
        points = QUARTIC_BEZIER[:n_points]
        for t, split_matrix in degree_dict.items():
            nt.assert_allclose(
                split_bezier(points, t), split_matrix @ points, atol=1e-15
            )

    for degree, degree_dict in SUBDIVISION_MATRICES.items():
        n_points = degree + 1
        points = QUARTIC_BEZIER[:n_points]
        # Split in half
        split_matrix = degree_dict[2]
        nt.assert_allclose(
            split_bezier(points, 0.5),
            split_matrix @ points,
        )


def test_get_subdivision_matrix() -> None:
    """Test that the memos in .:meth:`_get_subdivision_matrix`
    are being correctly generated.
    """
    # Only for degrees up to 3!
    for degree in range(4):
        degree_dict = SUBDIVISION_MATRICES[degree]
        for n_divisions, subdivision_matrix in degree_dict.items():
            nt.assert_allclose(
                _get_subdivision_matrix(degree + 1, n_divisions),
                subdivision_matrix,
            )


def test_subdivide_bezier() -> None:
    """Test that :func:`subdivide_bezier`, both in the memoized cases
    (degrees up to 3) and the fallback algorithm (degree 4), works
    correctly.
    """
    for degree, degree_dict in SUBDIVISION_MATRICES.items():
        n_points = degree + 1
        points = QUARTIC_BEZIER[:n_points]
        for n_divisions, subdivision_matrix in degree_dict.items():
            nt.assert_allclose(
                subdivide_bezier(points, n_divisions),
                subdivision_matrix @ points,
            )


def test_get_smooth_cubic_bezier_handle_points() -> None:
    """Test that :func:`.get_smooth_cubic_bezier_handle_points` returns the
    correct handles, both for open and closed Bézier splines.
    """
    open_curve_corners = np.array(
        [
            [1, 1, 0],
            [-1, 1, 1],
            [-1, -1, 2],
            [1, -1, 1],
        ],
        dtype=ManimFloat,
    )
    h1, h2 = get_smooth_cubic_bezier_handle_points(open_curve_corners)
    assert np.allclose(
        h1,
        np.array(
            [
                [1 / 5, 11 / 9, 13 / 45],
                [-7 / 5, 5 / 9, 64 / 45],
                [-3 / 5, -13 / 9, 91 / 45],
            ]
        ),
    )
    assert np.allclose(
        h2,
        np.array(
            [
                [-3 / 5, 13 / 9, 26 / 45],
                [-7 / 5, -5 / 9, 89 / 45],
                [1 / 5, -11 / 9, 68 / 45],
            ]
        ),
    )

    closed_curve_corners = np.array(
        [
            [1, 1, 0],
            [-1, 1, 1],
            [-1, -1, 2],
            [1, -1, 1],
            [1, 1, 0],
        ],
        dtype=ManimFloat,
    )
    h1, h2 = get_smooth_cubic_bezier_handle_points(closed_curve_corners)
    assert np.allclose(
        h1,
        np.array(
            [
                [1 / 2, 3 / 2, 0],
                [-3 / 2, 1 / 2, 3 / 2],
                [-1 / 2, -3 / 2, 2],
                [3 / 2, -1 / 2, 1 / 2],
            ]
        ),
    )
    assert np.allclose(
        h2,
        np.array(
            [
                [-1 / 2, 3 / 2, 1 / 2],
                [-3 / 2, -1 / 2, 2],
                [1 / 2, -3 / 2, 3 / 2],
                [3 / 2, 1 / 2, 0],
            ]
        ),
    )
