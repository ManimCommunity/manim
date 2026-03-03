from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from manim import Circle, Square
from manim.mobject.geometry.boolean_ops import _BooleanOps

if TYPE_CHECKING:
    from manim.mobject.types.vectorized_mobject import VMobject
    from manim.typing import Point2D_Array, Point3D_Array


@pytest.mark.parametrize(
    ("test_input", "expected"),
    [
        (
            [(1.0, 2.0), (3.0, 4.0)],
            [
                np.array([1.0, 2.0, 0]),
                np.array([3.0, 4.0, 0]),
            ],
        ),
        (
            [(1.1, 2.2)],
            [
                np.array([1.1, 2.2, 0.0]),
            ],
        ),
    ],
)
def test_convert_2d_to_3d_array(
    test_input: Point2D_Array, expected: Point3D_Array
) -> None:
    a = _BooleanOps()
    result = a._convert_2d_to_3d_array(test_input)
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert (result[i] == expected[i]).all()


def test_convert_2d_to_3d_array_zdim() -> None:
    a = _BooleanOps()
    result = a._convert_2d_to_3d_array([(1.0, 2.0)], z_dim=1.0)
    assert (result[0] == np.array([1.0, 2.0, 1.0])).all()


@pytest.mark.parametrize(
    "test_input",
    [
        Square(),
        Circle(),
        Square(side_length=4),
        Circle(radius=3),
    ],
)
def test_vmobject_to_skia_path_and_inverse(test_input: VMobject) -> None:
    a = _BooleanOps()
    path = a._convert_vmobject_to_skia_path(test_input)
    assert len(list(path.segments)) > 1

    new_vmobject = a._convert_skia_path_to_vmobject(path)
    # For some reason, there are 3 more points in the new VMobject than in the
    # original input.
    np.testing.assert_allclose(new_vmobject.points[:-3], test_input.points)
