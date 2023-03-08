from __future__ import annotations

import pytest

from manim import Circle, Mobject, Rectangle, Square, VGroup


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

    assert vg.width == 14.0, f"{vg.width}"
    assert vg.height == 20.0, f"{vg.height}"
    assert is_close(vg.depth, 0.9), f"{vg.depth}"

    # Dimensions should be recalculated after scaling
    vg.scale(0.5)
    assert vg.width == 7.0, f"{vg.width}"
    assert vg.height == 10.0, f"{vg.height}"
    assert is_close(vg.depth, 0.45), f"{vg.depth}"

    # Adding a mobject changes the bounds/dimensions
    rect = Rectangle(width=3, height=5)
    rect.move_to([9, 3, 1])
    vg += rect
    assert vg.width == 13.0, f"{vg.width}"
    assert is_close(vg.height, 18.5), f"{vg.height}"
    assert is_close(vg.depth, 0.775), f"{vg.depth}"
