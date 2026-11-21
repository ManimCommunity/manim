from __future__ import annotations

import copy

import numpy as np
import pytest

from manim import RIGHT, Mobject, Square
from manim.mobject.fields import ArrayField, Field, field_version
from manim.utils.hashing import get_json


def test_version_changes_on_assignment_only():
    square = Square()
    version = field_version(square, "points")

    square.get_center()
    square.set_fill(opacity=0.5)
    assert field_version(square, "points") == version

    square.shift(RIGHT)
    assert field_version(square, "points") != version


def test_points_are_stored_as_a_float_copy():
    square = Square()
    corners = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    square.points = corners
    corners[0] = [5, 5, 5]
    np.testing.assert_array_equal(square.points[0], [0, 0, 0])

    square.points = [[0, 0, 0]]
    assert square.points.dtype == np.float64


@pytest.mark.parametrize(
    "modify",
    [
        lambda points: points.__setitem__(0, [1, 2, 3]),
        lambda points: points.__iadd__(RIGHT),
        lambda points: points[1:].__setitem__(0, [1, 2, 3]),
        lambda points: points.T.__setitem__(0, 1),
        lambda points: points.reshape(-1).__setitem__(0, 1),
        lambda points: setattr(points.flags, "writeable", True),
    ],
    ids=["item", "in-place add", "slice", "transpose", "reshape", "writeable flag"],
)
def test_points_cannot_be_written_in_place(modify):
    square = Square()
    with pytest.raises(ValueError):
        modify(square.points)


@pytest.mark.parametrize(
    "alias",
    [
        lambda array: (array, array),
        lambda array: (array[:], array),
        lambda array: (array.view(), array),
    ],
    ids=["array", "slice", "view"],
)
def test_writing_to_an_assigned_array_does_not_change_the_field(alias):
    square = Square()
    assigned, writable = alias(np.zeros((4, 3)))

    square.points = assigned
    version = field_version(square, "points")
    writable[0] = [1, 2, 3]

    np.testing.assert_array_equal(square.points, np.zeros((4, 3)))
    assert field_version(square, "points") == version


def test_a_read_only_view_of_a_writable_array_is_copied():
    square = Square()
    data = np.zeros((4, 3))
    view = data.view()
    view.flags.writeable = False

    square.points = view
    data[0] = [1, 2, 3]

    np.testing.assert_array_equal(square.points, np.zeros((4, 3)))


def test_field_arrays_are_shared_without_a_copy():
    square, other = Square(), Square()

    other.points = square.points
    assert other.points is square.points
    assert square.copy().points is square.points


def test_versions_do_not_change_play_cache_keys():
    square, reassigned = Square(), Square()
    reassigned.points = reassigned.points + 0

    assert field_version(reassigned, "points") != field_version(square, "points")
    assert get_json(reassigned) == get_json(square)


def test_plain_field_has_version_zero_until_assigned():
    class Holder:
        width = Field()

    holder = Holder()
    assert field_version(holder, "width") == 0

    holder.width = 2.0
    assert holder.width == 2.0
    assert field_version(holder, "width") != 0


def test_a_shallow_copy_has_its_own_versions():
    square = Square()
    shallow = copy.copy(square)
    version = field_version(square, "points")

    shallow.shift(RIGHT)

    assert field_version(square, "points") == version
    assert field_version(shallow, "points") != version


def test_replacing_a_field_by_a_class_attribute_raises():
    with pytest.raises(TypeError, match=r"Shifted\.points replaces the field"):

        class Shifted(Mobject):
            points = np.zeros((1, 3))


def test_redefining_a_field_in_a_subclass_is_allowed():
    class Precise(Mobject):
        points: ArrayField[np.ndarray] = ArrayField(np.float64)

    assert Precise().points.dtype == np.float64
