from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from manim import (
    DL,
    DR,
    PI,
    UL,
    UR,
    Circle,
    Mobject,
    Rectangle,
    Square,
    Triangle,
    VGroup,
    VMobject,
)


def test_mobject_add():
    """Test Mobject.add()."""
    """Call this function with a Container instance to test its add() method."""
    # check that obj.submobjects is updated correctly
    obj = Mobject()
    assert len(obj.submobjects) == 0
    obj.add(Mobject())
    assert len(obj.submobjects) == 1
    obj.add(*(Mobject() for _ in range(10)))
    assert len(obj.submobjects) == 11

    # check that adding a mobject twice does not actually add it twice
    repeated = Mobject()
    obj.add(repeated)
    assert len(obj.submobjects) == 12
    obj.add(repeated)
    assert len(obj.submobjects) == 12

    # check that Mobject.add() returns the Mobject (for chained calls)
    assert obj.add(Mobject()) is obj
    assert len(obj.submobjects) == 13

    obj = Mobject()

    # a Mobject cannot contain itself
    with pytest.raises(ValueError) as add_self_info:
        obj.add(Mobject(), obj, Mobject())
    assert str(add_self_info.value) == (
        "Cannot add Mobject as a submobject of itself (at index 1)."
    )
    assert len(obj.submobjects) == 0

    # can only add Mobjects
    with pytest.raises(TypeError) as add_str_info:
        obj.add(Mobject(), Mobject(), "foo")
    assert str(add_str_info.value) == (
        "Only values of type Mobject can be added as submobjects of Mobject, "
        "but the value foo (at index 2) is of type str."
    )
    assert len(obj.submobjects) == 0


def test_mobject_remove():
    """Test Mobject.remove()."""
    obj = Mobject()
    to_remove = Mobject()
    obj.add(to_remove)
    obj.add(*(Mobject() for _ in range(10)))
    assert len(obj.submobjects) == 11
    obj.remove(to_remove)
    assert len(obj.submobjects) == 10
    obj.remove(to_remove)
    assert len(obj.submobjects) == 10

    assert obj.remove(Mobject()) is obj


def test_mobject_dimensions_single_mobject():
    # A Mobject with no points and no submobjects has no dimensions
    empty = Mobject()
    assert empty.width == 0
    assert empty.height == 0
    assert empty.depth == 0

    has_points = Mobject()
    has_points.points = np.array([[-1, -2, -3], [1, 3, 5]])
    assert has_points.width == 2
    assert has_points.height == 5
    assert has_points.depth == 8

    rect = Rectangle(width=3, height=5)

    assert rect.width == 3
    assert rect.height == 5
    assert rect.depth == 0

    # Dimensions should be recalculated after scaling
    rect.scale(2.0)
    assert rect.width == 6
    assert rect.height == 10
    assert rect.depth == 0

    # Dimensions should not be dependent on location
    rect.move_to([-3, -4, -5])
    assert rect.width == 6
    assert rect.height == 10
    assert rect.depth == 0

    circ = Circle(radius=2)

    assert circ.width == 4
    assert circ.height == 4
    assert circ.depth == 0


def is_close(x, y):
    return abs(x - y) < 0.00001


def test_mobject_dimensions_nested_mobjects():
    vg = VGroup()

    for x in range(-5, 8, 1):
        row = VGroup()
        vg += row
        for y in range(-17, 2, 1):
            for z in range(0, 10, 1):
                s = Square().move_to([x, y, z / 10])
                row += s

    assert vg.width == 14.0, vg.width
    assert vg.height == 20.0, vg.height
    assert is_close(vg.depth, 0.9), vg.depth

    # Dimensions should be recalculated after scaling
    vg.scale(0.5)
    assert vg.width == 7.0, vg.width
    assert vg.height == 10.0, vg.height
    assert is_close(vg.depth, 0.45), vg.depth

    # Adding a mobject changes the bounds/dimensions
    rect = Rectangle(width=3, height=5)
    rect.move_to([9, 3, 1])
    vg += rect
    assert vg.width == 13.0, vg.width
    assert is_close(vg.height, 18.5), vg.height
    assert is_close(vg.depth, 0.775), vg.depth


def test_mobject_dimensions_mobjects_with_no_points():
    empty_mob = VMobject()
    assert empty_mob.width == 0
    assert empty_mob.height == 0

    for direction in [DL, DR, UL, UR]:
        rect = Rectangle(width=2, height=3)
        rect.move_to(direction * 10)
        outer_group = VGroup(rect)

        # This is as one would expect
        assert outer_group.width == 2
        assert outer_group.height == 3

        # Adding a submobject with no points does not change the group size
        outer_group.add(empty_mob)
        assert outer_group.width == 2
        assert outer_group.height == 3


def test_mobject_dimensions_has_points_and_children():
    outer_rect = Rectangle(width=3, height=6)
    inner_rect = Rectangle(width=2, height=1)
    inner_rect.align_to(outer_rect.get_corner(UR), DL)
    outer_rect.add(inner_rect)

    # The width of a mobject should depend both on its points and
    # the points of all children mobjects.
    assert outer_rect.width == 5  # 3 from outer_rect, 2 from inner_rect
    assert outer_rect.height == 7  # 6 from outer_rect, 1 from inner_rect
    assert outer_rect.depth == 0

    assert inner_rect.width == 2
    assert inner_rect.height == 1
    assert inner_rect.depth == 0


def test_rotate_about_vertex_view():
    """Test that rotating about a vertex obtained from get_vertices() works correctly.

    This is a regression test for an issue where get_vertices() returns a view of the points array,
    and using it as about_point in rotate() would cause the view to be mutated.
    """
    triangle = Triangle()
    original_vertices = triangle.get_vertices().copy()
    first_vertex = original_vertices[0].copy()

    # This should rotate about the first vertex without corrupting it
    triangle.rotate(PI / 2, about_point=triangle.get_vertices()[0])

    # The first vertex should remain in the same position (within numerical precision)
    rotated_vertices = triangle.get_vertices()
    np.testing.assert_allclose(rotated_vertices[0], first_vertex, atol=1e-6)


def test_scale_about_vertex_view():
    """Test that scaling about a vertex obtained from get_vertices() works correctly.

    This is a regression test for an issue where get_vertices() returns a view of the points array,
    and using it as about_point in scale() would cause the view to be mutated.
    """
    triangle = Triangle()
    original_vertices = triangle.get_vertices().copy()
    first_vertex = original_vertices[0].copy()

    # This should scale about the first vertex without corrupting it
    triangle.scale(2, about_point=triangle.get_vertices()[0])

    # The first vertex should remain in the same position (within numerical precision)
    scaled_vertices = triangle.get_vertices()
    np.testing.assert_allclose(scaled_vertices[0], first_vertex, atol=1e-6)


def test_stretch_about_vertex_view():
    """Test that stretching about a vertex obtained from get_vertices() works correctly.

    This is a regression test for an issue where get_vertices() returns a view of the points array,
    and using it as about_point in stretch() would cause the view to be mutated.
    """
    triangle = Triangle()
    original_vertices = triangle.get_vertices().copy()
    first_vertex = original_vertices[0].copy()

    # This should stretch about the first vertex without corrupting it
    triangle.stretch(2, 0, about_point=triangle.get_vertices()[0])

    # The first vertex should remain in the same position (within numerical precision)
    stretched_vertices = triangle.get_vertices()
    np.testing.assert_allclose(stretched_vertices[0], first_vertex, atol=1e-6)


def test_apply_matrix_about_vertex_view():
    """Test that apply_matrix about a vertex obtained from get_vertices() works correctly.

    This is a regression test for an issue where get_vertices() returns a view of the points array,
    and using it as about_point in apply_matrix() would cause the view to be mutated.
    """
    triangle = Triangle()
    original_vertices = triangle.get_vertices().copy()
    first_vertex = original_vertices[0].copy()

    # Define a rotation matrix (90 degrees rotation around z-axis)
    rotation_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # This should apply the matrix about the first vertex without corrupting it
    triangle.apply_matrix(rotation_matrix, about_point=triangle.get_vertices()[0])

    # The first vertex should remain in the same position (within numerical precision)
    transformed_vertices = triangle.get_vertices()
    np.testing.assert_allclose(transformed_vertices[0], first_vertex, atol=1e-6)


# Tests for methods which provide Mobject's sequence interface


def build_mobject(n=10, has_points=False, mob_cls=Mobject):
    """Construct a mobject (with/without points) with n submobjects, and return it along with its
    split() result.
    """
    mob = mob_cls()
    mob.add(*[mob_cls() for _ in range(n)])
    if has_points:
        mob.points = np.zeros((3, 3))
    return mob, mob.split()


@pytest.mark.parametrize("has_points", [True, False])
def test_getitem_single_index(has_points):
    """Test that indexing a mobject behaves the same as indexing Mobject.split()."""
    n = 10
    mob, mob_list = build_mobject(n, has_points=has_points)
    length = (n + 1) if has_points else n
    for i in range(-length, length):
        assert mob[i] == mob_list[i]

    # Check that indexing outside valid range raises IndexError
    with pytest.raises(IndexError):
        mob[length]
    with pytest.raises(IndexError):
        mob[-length - 1]


@pytest.mark.parametrize("has_points", [True, False])
def test_getitem_single_index_no_submobjects(has_points):
    """Test that indexing into a mobject with no submobjects behaves correctly."""
    mob, _ = build_mobject(0, has_points=has_points)
    if has_points:
        assert mob[0] == mob
    else:
        with pytest.raises(IndexError):
            mob[0]
    with pytest.raises(IndexError):
        mob[1]


@pytest.mark.parametrize("has_points", [True, False])
def test_getitem_slice(has_points):
    """Test that slicing a mobject behaves the same as slicing Mobject.split()."""
    n = 10
    mob, mob_list = build_mobject(n, has_points=has_points)
    length = (n + 1) if has_points else n
    for start, stop, step in product(range(-length, length), repeat=3):
        assert mob[start:stop].submobjects == mob_list[start:stop]
        if step == 0:
            with pytest.raises(ValueError):
                mob[start:stop:step]
        else:
            assert mob[start:stop:step].submobjects == mob_list[start:stop:step]

    # Check that slicing with too high stop value is possible
    too_high_index = length * 2
    assert mob[1:too_high_index].submobjects == mob[1:].submobjects
    assert mob[1:too_high_index].submobjects == mob_list[1:too_high_index]

    # Check that slicing with too low start value works
    too_low_index = -(length * 2)
    assert mob[too_low_index:1].submobjects == mob[:1].submobjects
    assert mob[too_low_index:1].submobjects == mob_list[too_low_index:1]


@pytest.mark.parametrize("has_points", [True, False])
@pytest.mark.parametrize("mob_cls", [Mobject, VMobject])
def test_getitem_slice_group_type(has_points, mob_cls):
    """Test that the group created from slicing a mobject has the appropriate type
    for the type of the mobject.
    """
    n = 10
    mob, _ = build_mobject(n, has_points=has_points, mob_cls=mob_cls)
    group_cls = mob.get_group_class()
    length = (n + 1) if has_points else n
    for start, stop, step in product(range(-length, length), repeat=3):
        if not step:
            continue
        assert type(mob[start:stop:step]) is group_cls


@pytest.mark.parametrize("has_points", [True, False])
def test_getitem_slice_no_submobjects(has_points):
    """Test that slicing a mobject with no submobjects behaves correctly."""
    mob, _ = build_mobject(0, has_points=has_points)
    expected = [] if not has_points else [mob]
    assert mob[:].submobjects == expected
    assert mob[::].submobjects == expected
    assert mob[:1].submobjects == expected
    assert mob[::2].submobjects == expected


@pytest.mark.parametrize("has_points", [True, False])
def test_getitem_index_types(has_points):
    """Test that __getitem__ accepts anything implementing __index__ and rejects
    floats, like the list type does.
    """
    n = 10
    mob, mob_list = build_mobject(n, has_points=has_points)

    class Ix:
        def __index__(self):
            return 1

    assert mob[0] == mob_list[0]
    assert mob[True] == mob_list[1]
    assert mob[np.int64(2)] == mob_list[2]  # e.g. mob[np.argmin(...)]
    assert mob[Ix()] == mob_list[1]
    with pytest.raises(TypeError):
        mob[1.5]

    if not has_points:
        # Indexing into an empty mobject with an invalid type should throw
        # a TypeError instead of IndexError
        with pytest.raises(TypeError):
            mob[1.5]


@pytest.mark.parametrize("has_points", [True, False])
@pytest.mark.parametrize("n", [0, 10])
def test_iter_mobject_unpacking(has_points, n):
    """Test that unpacking a Mobject behaves correctly."""
    mob, mob_list = build_mobject(n, has_points=has_points)
    assert list(mob) == mob_list
    assert [*mob] == mob_list


@pytest.mark.parametrize("has_points", [True, False])
@pytest.mark.parametrize("n", [0, 10])
def test_iter_mobject_iteration(has_points, n):
    """Test that iterating over a Mobject behaves the same as iterating over Mobject.split()."""
    mob, mob_list = build_mobject(n, has_points=has_points)
    mob_iterator = iter(mob)
    mob_list_iterator = iter(mob_list)
    for a, b in zip(mob_iterator, mob_list_iterator, strict=True):
        assert a == b


@pytest.mark.parametrize("has_points", [True, False])
@pytest.mark.parametrize("n", [0, 10])
def test_len_mobject(has_points, n):
    """Test that len(mob) behaves the same as len(mob.split())."""
    mob, mob_list = build_mobject(n, has_points=has_points)
    assert len(mob) == len(mob_list)
