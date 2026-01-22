import numpy as np
import pytest

from manim.utils.polylabel import Cell, Polygon, polylabel


# Test simple square and square with a hole for inside/outside logic
@pytest.mark.parametrize(
    ("rings", "inside_points", "outside_points"),
    [
        (
            # Simple square: basic convex polygon
            [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],  # rings
            [
                [2, 2],
                [1, 1],
                [3.9, 3.9],
                [0, 0],
                [2, 0],
                [0, 2],
                [0, 4],
                [4, 0],
                [4, 2],
                [2, 4],
                [4, 4],
            ],  # inside points
            [[-1, -1], [5, 5], [4.1, 2]],  # outside points
        ),
        (
            # Square with a square hole (donut shape): tests handling of interior voids
            [
                [[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]],
                [[2, 2], [2, 4], [4, 4], [4, 2], [2, 2]],
            ],  # rings
            [[1.5, 1.5], [3, 1.5], [1.5, 3]],  # inside points
            [[3, 3], [6, 6], [0, 0]],  # outside points
        ),
        (
            # Non-convex polygon (same shape as flags used in Brazilian june festivals)
            [[[0, 0], [2, 2], [4, 0], [4, 4], [0, 4], [0, 0]]],  # rings
            [[1, 3], [3.9, 3.9], [2, 3.5]],  # inside points
            [
                [0.1, 0],
                [1, 0],
                [2, 0],
                [2, 1],
                [2, 1.9],
                [3, 0],
                [3.9, 0],
            ],  # outside points
        ),
    ],
)
def test_polygon_inside_outside(rings, inside_points, outside_points):
    polygon = Polygon(rings)
    for point in inside_points:
        assert polygon.inside(point)

    for point in outside_points:
        assert not polygon.inside(point)


# Test distance calculation with known expected distances
@pytest.mark.parametrize(
    ("rings", "points", "expected_distance"),
    [
        (
            [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],  # rings
            [[2, 2]],  # points
            2.0,  # Distance from center to closest edge in square
        ),
        (
            [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],  # rings
            [[0, 0], [2, 0], [4, 2], [2, 4], [0, 2]],  # points
            0.0,  # On the edge
        ),
        (
            [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],  # rings
            [[5, 5]],  # points
            -np.sqrt(2),  # Outside and diagonally offset
        ),
    ],
)
def test_polygon_compute_distance(rings, points, expected_distance):
    polygon = Polygon(rings)
    for point in points:
        result = polygon.compute_distance(np.array(point))
        assert pytest.approx(result, rel=1e-3) == expected_distance


@pytest.mark.parametrize(
    ("center", "h", "rings"),
    [
        (
            [2, 2],  # center
            1.0,  # h
            [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],  # rings
        ),
        (
            [3, 1.5],  # center
            0.5,  # h
            [
                [[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]],
                [[2, 2], [2, 4], [4, 4], [4, 2], [2, 2]],
            ],  # rings
        ),
    ],
)
def test_cell(center, h, rings):
    polygon = Polygon(rings)
    cell = Cell(center, h, polygon)
    assert isinstance(cell.d, float)
    assert isinstance(cell.p, float)
    assert np.allclose(cell.c, center)
    assert cell.h == h

    other = Cell(np.add(center, [0.1, 0.1]), h, polygon)
    assert (cell < other) == (cell.d < other.d)
    assert (cell > other) == (cell.d > other.d)
    assert (cell <= other) == (cell.d <= other.d)
    assert (cell >= other) == (cell.d >= other.d)


@pytest.mark.parametrize(
    ("rings", "expected_centers"),
    [
        (
            # Simple square: basic convex polygon
            [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],
            [[2.0, 2.0]],  # single correct pole of inaccessibility
        ),
        (
            # Square with a square hole (donut shape): tests handling of interior voids
            [
                [[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]],
                [[2, 2], [2, 4], [4, 4], [4, 2], [2, 2]],
            ],
            [  # any of the four pole of inaccessibility options
                [1.5, 1.5],
                [1.5, 4.5],
                [4.5, 1.5],
                [4.5, 4.5],
            ],
        ),
    ],
)
def test_polylabel(rings, expected_centers):
    # Add third dimension to conform to polylabel input format
    rings_3d = [np.column_stack([ring, np.zeros(len(ring))]) for ring in rings]
    result = polylabel(rings_3d, precision=0.01)

    assert isinstance(result, Cell)
    assert result.h <= 0.01
    assert result.d >= 0.0

    match_found = any(np.allclose(result.c, ec, atol=0.1) for ec in expected_centers)
    assert match_found, f"Expected one of {expected_centers}, but got {result.c}"
