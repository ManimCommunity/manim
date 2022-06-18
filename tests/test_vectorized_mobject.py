from math import cos, sin

import numpy as np
import pytest

from manim import (
    Circle,
    Line,
    Mobject,
    Polygon,
    RegularPolygon,
    Square,
    VDict,
    VGroup,
    VMobject,
)
from manim.constants import PI


def test_vmobject_point_from_propotion():
    obj = VMobject()

    # One long line, one short line
    obj.set_points_as_corners(
        [
            np.array([0, 0, 0]),
            np.array([4, 0, 0]),
            np.array([4, 2, 0]),
        ],
    )

    # Total length of 6, so halfway along the object
    # would be at length 3, which lands in the first, long line.
    np.testing.assert_array_equal(obj.point_from_proportion(0.5), np.array([3, 0, 0]))

    with pytest.raises(ValueError, match="between 0 and 1"):
        obj.point_from_proportion(2)

    obj.clear_points()
    with pytest.raises(Exception, match="with no points"):
        obj.point_from_proportion(0)


def test_vgroup_init():
    """Test the VGroup instantiation."""
    VGroup()
    VGroup(VMobject())
    VGroup(VMobject(), VMobject())
    with pytest.raises(TypeError):
        VGroup(Mobject())
    with pytest.raises(TypeError):
        VGroup(Mobject(), Mobject())


def test_vgroup_add():
    """Test the VGroup add method."""
    obj = VGroup()
    assert len(obj.submobjects) == 0
    obj.add(VMobject())
    assert len(obj.submobjects) == 1
    with pytest.raises(TypeError):
        obj.add(Mobject())
    assert len(obj.submobjects) == 1
    with pytest.raises(TypeError):
        # If only one of the added object is not an instance of VMobject, none of them should be added
        obj.add(VMobject(), Mobject())
    assert len(obj.submobjects) == 1
    with pytest.raises(ValueError):
        # a Mobject cannot contain itself
        obj.add(obj)


def test_vgroup_add_dunder():
    """Test the VGroup __add__ magic method."""
    obj = VGroup()
    assert len(obj.submobjects) == 0
    obj + VMobject()
    assert len(obj.submobjects) == 0
    obj += VMobject()
    assert len(obj.submobjects) == 1
    with pytest.raises(TypeError):
        obj += Mobject()
    assert len(obj.submobjects) == 1
    with pytest.raises(TypeError):
        # If only one of the added object is not an instance of VMobject, none of them should be added
        obj += (VMobject(), Mobject())
    assert len(obj.submobjects) == 1
    with pytest.raises(ValueError):
        # a Mobject cannot contain itself
        obj += obj


def test_vgroup_remove():
    """Test the VGroup remove method."""
    a = VMobject()
    c = VMobject()
    b = VGroup(c)
    obj = VGroup(a, b)
    assert len(obj.submobjects) == 2
    assert len(b.submobjects) == 1
    obj.remove(a)
    b.remove(c)
    assert len(obj.submobjects) == 1
    assert len(b.submobjects) == 0
    obj.remove(b)
    assert len(obj.submobjects) == 0


def test_vgroup_remove_dunder():
    """Test the VGroup __sub__ magic method."""
    a = VMobject()
    c = VMobject()
    b = VGroup(c)
    obj = VGroup(a, b)
    assert len(obj.submobjects) == 2
    assert len(b.submobjects) == 1
    assert len(obj - a) == 1
    assert len(obj.submobjects) == 2
    obj -= a
    b -= c
    assert len(obj.submobjects) == 1
    assert len(b.submobjects) == 0
    obj -= b
    assert len(obj.submobjects) == 0


def test_vmob_add_to_back():
    """Test the Mobject add_to_back method."""
    a = VMobject()
    b = Line()
    c = "text"
    with pytest.raises(ValueError):
        # Mobject cannot contain self
        a.add_to_back(a)
    with pytest.raises(TypeError):
        # All submobjects must be of type Mobject
        a.add_to_back(c)

    # No submobject gets added twice
    a.add_to_back(b)
    a.add_to_back(b, b)
    assert len(a.submobjects) == 1
    a.submobjects.clear()
    a.add_to_back(b, b, b)
    a.add_to_back(b, b)
    assert len(a.submobjects) == 1
    a.submobjects.clear()

    # Make sure the ordering has not changed
    o1, o2, o3 = Square(), Line(), Circle()
    a.add_to_back(o1, o2, o3)
    assert a.submobjects.pop() == o3
    assert a.submobjects.pop() == o2
    assert a.submobjects.pop() == o1


def test_vdict_init():
    """Test the VDict instantiation."""
    # Test empty VDict
    VDict()
    # Test VDict made from list of pairs
    VDict([("a", VMobject()), ("b", VMobject()), ("c", VMobject())])
    # Test VDict made from a python dict
    VDict({"a": VMobject(), "b": VMobject(), "c": VMobject()})
    # Test VDict made using zip
    VDict(zip(["a", "b", "c"], [VMobject(), VMobject(), VMobject()]))
    # If the value is of type Mobject, must raise a TypeError
    with pytest.raises(TypeError):
        VDict({"a": Mobject()})


def test_vdict_add():
    """Test the VDict add method."""
    obj = VDict()
    assert len(obj.submob_dict) == 0
    obj.add([("a", VMobject())])
    assert len(obj.submob_dict) == 1
    with pytest.raises(TypeError):
        obj.add([("b", Mobject())])


def test_vdict_remove():
    """Test the VDict remove method."""
    obj = VDict([("a", VMobject())])
    assert len(obj.submob_dict) == 1
    obj.remove("a")
    assert len(obj.submob_dict) == 0
    with pytest.raises(KeyError):
        obj.remove("a")


def test_vgroup_supports_item_assigment():
    """Test VGroup supports array-like assignment for VMObjects"""
    a = VMobject()
    b = VMobject()
    vgroup = VGroup(a)
    assert vgroup[0] == a
    vgroup[0] = b
    assert vgroup[0] == b
    assert len(vgroup) == 1


def test_vgroup_item_assignment_at_correct_position():
    """Test VGroup item-assignment adds to correct position for VMObjects"""
    n_items = 10
    vgroup = VGroup()
    for _i in range(n_items):
        vgroup.add(VMobject())
    new_obj = VMobject()
    vgroup[6] = new_obj
    assert vgroup[6] == new_obj
    assert len(vgroup) == n_items


def test_vgroup_item_assignment_only_allows_vmobjects():
    """Test VGroup item-assignment raises TypeError when invalid type is passed"""
    vgroup = VGroup(VMobject())
    with pytest.raises(TypeError, match="All submobjects must be of type VMobject"):
        vgroup[0] = "invalid object"


def test_trim_dummy():
    o = VMobject()
    o.start_new_path(np.array([0, 0, 0]))
    o.add_line_to(np.array([1, 0, 0]))
    o.add_line_to(np.array([2, 0, 0]))
    o.add_line_to(np.array([2, 0, 0]))  # Dummy point, will be stripped from points
    o.start_new_path(np.array([0, 1, 0]))
    o.add_line_to(np.array([1, 2, 0]))

    o2 = VMobject()
    o2.start_new_path(np.array([0, 0, 0]))
    o2.add_line_to(np.array([0, 1, 0]))
    o2.start_new_path(np.array([1, 0, 0]))
    o2.add_line_to(np.array([1, 1, 0]))
    o2.add_line_to(np.array([1, 2, 0]))

    def path_length(p):
        return len(p) // o.n_points_per_cubic_curve

    assert tuple(map(path_length, o.get_subpaths())) == (3, 1)
    assert tuple(map(path_length, o2.get_subpaths())) == (1, 2)

    o.align_points(o2)

    assert tuple(map(path_length, o.get_subpaths())) == (2, 2)
    assert tuple(map(path_length, o2.get_subpaths())) == (2, 2)


def test_bounded_become():
    """Tests that align_points generates a bounded number of points.
    https://github.com/ManimCommunity/manim/issues/1959
    """
    o = VMobject()

    def draw_circle(m: VMobject, n_points, x=0, y=0, r=1):
        center = np.array([x, y, 0])
        m.start_new_path(center + [r, 0, 0])
        for i in range(1, n_points + 1):
            theta = 2 * PI * i / n_points
            m.add_line_to(center + [cos(theta) * r, sin(theta) * r, 0])

    # o must contain some points, or else become behaves differently
    draw_circle(o, 2)

    for _ in range(20):
        # Alternate between calls to become with different subpath sizes
        a = VMobject()
        draw_circle(a, 20)
        o.become(a)
        b = VMobject()
        draw_circle(b, 15)
        draw_circle(b, 15, x=3)
        o.become(b)

    # The number of points should be similar to the size of a and b
    assert len(o.points) <= (20 + 15 + 15) * 4


def test_vmobject_same_points_become():
    a = Square()
    b = Circle()
    a.become(b)
    np.testing.assert_array_equal(a.points, b.points)
    assert len(a.submobjects) == len(b.submobjects)


def test_vmobject_same_num_submobjects_become():
    a = Square()
    b = RegularPolygon(n=6)
    a.become(b)
    np.testing.assert_array_equal(a.points, b.points)
    assert len(a.submobjects) == len(b.submobjects)


def test_vmobject_different_num_points_and_submobjects_become():
    a = Square()
    b = VGroup(Circle(), Square())
    a.become(b)
    np.testing.assert_array_equal(a.points, b.points)
    assert len(a.submobjects) == len(b.submobjects)


def test_vmobject_point_at_angle():
    a = Circle()
    p = a.point_at_angle(4 * PI)
    np.testing.assert_array_equal(a.points[0], p)


def test_proportion_from_point():
    A = np.sqrt(3) * np.array([0, 1, 0])
    B = np.array([-1, 0, 0])
    C = np.array([1, 0, 0])
    abc = Polygon(A, B, C)
    abc.shift(np.array([-1, 0, 0]))
    abc.scale(0.8)
    props = [abc.proportion_from_point(p) for p in abc.get_vertices()]
    np.testing.assert_allclose(props, [0, 1 / 3, 2 / 3])
