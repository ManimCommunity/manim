from __future__ import annotations

import numpy as np
import pytest

from manim import DL, UR, Circle, Mobject, Rectangle, Square, VGroup


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
    obj = Mobject()

    # a Mobject cannot contain itself
    with pytest.raises(ValueError):
        obj.add(obj)

    # can only add Mobjects
    with pytest.raises(TypeError):
        obj.add("foo")


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


def test_mobject_dimensions_mobjects_with_no_points_are_at_origin():
    rect = Rectangle(width=2, height=3)
    rect.move_to([-4, -5, 0])
    outer_group = VGroup(rect)

    # This is as one would expect
    assert outer_group.width == 2
    assert outer_group.height == 3

    # Adding a mobject with no points has a quirk of adding a "point"
    # to [0, 0, 0] (the origin). This changes the size of the outer
    # group because now the bottom left corner is at [-5, -6.5, 0]
    # but the upper right corner is [0, 0, 0] instead of [-3, -3.5, 0]
    outer_group.add(VGroup())
    assert outer_group.width == 5
    assert outer_group.height == 6.5


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
