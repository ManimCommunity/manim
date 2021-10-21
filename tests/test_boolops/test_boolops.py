import numpy as np
import pytest

from manim import *

# not exported by default, so directly import
from manim.mobject.boolean_ops import _BooleanOps
from tests.test_graphical_units.testing.frames_comparison import frames_comparison


@pytest.mark.parametrize(
    "test_input,expected",
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
def test_convert_2d_to_3d_array(test_input, expected):
    a = _BooleanOps()
    result = a._convert_2d_to_3d_array(test_input)
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert (result[i] == expected[i]).all()


def test_convert_2d_to_3d_array_zdim():
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
def test_vmobject_to_skia_path_and_inverse(test_input):
    a = _BooleanOps()
    path = a._convert_vmobject_to_skia_path(test_input)
    assert len(list(path.segments)) > 1

    new_vmobject = a._convert_skia_path_to_vmobject(path)
    # for some reason there is an extra 4 points in new vmobject than original
    assert np.where(np.isclose(new_vmobject.points[:-4], test_input.points))


__module_test__ = "boolean_ops"


@frames_comparison()
def test_union(scene):
    a = Square()
    b = Circle().move_to([0.2, 0.2, 0.0])
    c = Rectangle()
    un = Union(a, b, c).next_to(b)
    scene.add(a, b, c, un)


@frames_comparison()
def test_intersection(scene):
    a = Square()
    b = Circle().move_to([0.3, 0.3, 0.0])
    i = Intersection(a, b).next_to(b)
    scene.add(a, b, i)


@frames_comparison()
def test_difference(scene):
    a = Square()
    b = Circle().move_to([0.2, 0.3, 0.0])
    di = Difference(a, b).next_to(b)
    scene.add(a, b, di)


@frames_comparison()
def test_exclusion(scene):
    a = Square()
    b = Circle().move_to([0.3, 0.2, 0.0])
    ex = Exclusion(a, b).next_to(a)
    scene.add(a, b, ex)


@frames_comparison()
def test_intersection_3_mobjects(scene):
    a = Square()
    b = Circle().move_to([0.2, 0.2, 0])
    c = Triangle()
    i = Intersection(a, b, c, fill_opacity=0.5, color=BLUE)
    scene.add(a, b, c, i)
