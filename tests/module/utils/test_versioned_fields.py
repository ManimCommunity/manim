from __future__ import annotations

import copy

import numpy as np
import pytest

from manim import (
    BLUE,
    RED,
    RIGHT,
    Circle,
    ImageMobject,
    Mobject,
    Square,
    Transform,
    VMobject,
)
from manim.utils.hashing import get_json
from manim.utils.versioned_fields import ArrayField, Field, field_version


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


def test_the_newest_assignment_has_the_largest_version():
    class Holder:
        width = Field()
        height = Field()

    holder, other = Holder(), Holder()
    holder.width = 1.0
    holder.height = 1.0
    other.width = 1.0
    holder.width = 2.0

    versions = [field_version(holder, name) for name in ("width", "height")]
    assert max(versions) == field_version(holder, "width")
    assert field_version(holder, "width") > field_version(other, "width")


def test_an_assignment_takes_one_version(monkeypatch):
    import manim.utils.versioned_fields as versioned_fields

    taken = []
    next_version = versioned_fields._NEXT_VERSION
    monkeypatch.setattr(
        versioned_fields, "_NEXT_VERSION", lambda: taken.append(1) or next_version()
    )

    class Holder:
        width = Field()

    holder = Holder()
    holder.width = 1.0
    holder.width = 2.0
    assert len(taken) == 2


def test_a_shallow_copy_has_its_own_versions():
    square = Square()
    shallow = copy.copy(square)
    version = field_version(square, "points")

    shallow.shift(RIGHT)

    assert field_version(square, "points") == version
    assert field_version(shallow, "points") != version


@pytest.mark.parametrize("copy_function", [copy.copy, copy.deepcopy])
def test_copies_get_a_new_uid(copy_function):
    square = Square()
    assert copy_function(square).uid != square.uid
    assert Square().uid != square.uid


def test_paint_and_pixel_arrays_are_read_only():
    square = Square().set_fill(opacity=1)
    image = ImageMobject(np.zeros((2, 2, 4), dtype=np.uint8))

    for array in (square.fill.rgbas, square.stroke.rgbas, image.pixel_array):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1


def test_replacing_a_field_by_a_class_attribute_raises():
    with pytest.raises(TypeError, match=r"Shifted\.points replaces the field"):

        class Shifted(Mobject):
            points = np.zeros((1, 3))


def test_redefining_a_field_in_a_subclass_is_allowed():
    class Precise(Mobject):
        points: ArrayField[np.ndarray] = ArrayField(np.float64)

    assert Precise().points.dtype == np.float64


def test_replacing_a_style_field_by_a_class_attribute_raises():
    with pytest.raises(TypeError, match=r"Thick\.stroke_width replaces the field"):

        class Thick(Square):
            stroke_width = 20


def test_copies_share_only_values_that_cannot_change():
    square = Square()
    square.label = "corner"
    square.tags = ["a"]
    copied = square.copy()

    assert copied.label is square.label
    assert copied.tags == square.tags
    assert copied.tags is not square.tags


def test_svg_paths_share_their_parsed_path():
    import svgelements as se

    from manim.mobject.svg.svg_mobject import VMobjectFromSVGPath

    path = VMobjectFromSVGPath(se.Path("M 0 0 L 1 0 L 1 1 Z"))
    assert path.copy().path_obj is path.path_obj


def test_animation_steps_that_change_nothing_keep_versions():
    # A transform to a moved copy: the paint is the same at both ends, and keeps
    # its version over the whole animation.
    start = Square().set_color(RED)
    end = start.copy().shift(RIGHT)
    moving = start.copy()

    moving.interpolate(start, end, 0.3)
    version = field_version(moving, "fill")
    for alpha in (0.5, 0.8, 1):
        moving.interpolate(start, end, alpha)
    assert field_version(moving, "fill") == version

    moving.stroke_width = 2.5
    version = field_version(moving, "stroke_width")
    moving.stroke_width = float("2.5")  # equal, but another object
    assert field_version(moving, "stroke_width") == version

    partial = VMobject()
    partial.pointwise_become_partial(end, 0, 1)
    points_version = field_version(partial, "points")
    partial.pointwise_become_partial(end, 0, 1)
    assert field_version(partial, "points") == points_version


@pytest.mark.parametrize("alpha", [0, 1])
def test_animation_steps_call_a_path_that_does_not_meet_its_ends(alpha):
    # Only the straight path is taken to give its ends exactly; any other path
    # decides what is shown at the first and last frame (review of #5173).
    start, end = Square(), Circle()
    start.align_data(end)
    lifted = np.array([0.0, 0.0, 1.0])

    def path(a, b, t):
        return a + (b - a) * t + lifted

    moving = start.copy()
    moving.interpolate(start, end, alpha, path_func=path)

    np.testing.assert_allclose(moving.points, path(start.points, end.points, alpha))


def test_an_interpolated_value_cannot_be_written_through():
    start, end = Square(), Circle()
    start.align_data(end)
    moving = start.copy()

    moving.interpolate(start, end, 1)

    with pytest.raises(ValueError, match="read-only"):
        moving.points[0] = [9, 9, 9]


@pytest.mark.parametrize(
    ("change", "changed"),
    [
        (lambda m: m.shift(RIGHT), {"points"}),
        (lambda m: m.set_color(BLUE), {"fill", "stroke"}),
    ],
    ids=["moved", "recolored"],
)
def test_a_transform_gives_new_versions_only_to_the_fields_it_changes(change, changed):
    start = Square().set_fill(RED, 0.5)
    end = change(start.copy())
    animation = Transform(start, end)
    animation.begin()
    animation.interpolate(0.1)
    before = dict(start._field_versions)
    animation.interpolate(0.5)
    after = dict(start._field_versions)
    assert {name for name in after if after[name] != before.get(name)} == changed


def test_a_converted_field_stores_the_converted_value():
    from manim.utils.versioned_fields import ConvertedField

    class Holder:
        width = ConvertedField(lambda value: max(0.0, float(value)))

    holder = Holder()
    holder.width = -3
    assert holder.width == 0.0
    version = field_version(holder, "width")
    holder.width = -1  # converts to the same value: no new version
    assert field_version(holder, "width") == version
    holder.width = 2
    assert holder.width == 2.0
    assert field_version(holder, "width") != version
