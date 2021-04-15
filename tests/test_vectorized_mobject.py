import numpy as np
import pytest

from manim import Line, Mobject, VDict, VGroup, VMobject


def test_vmobject_point_from_propotion():
    obj = VMobject()

    # One long line, one short line
    obj.set_points_as_corners(
        [
            np.array([0, 0, 0]),
            np.array([4, 0, 0]),
            np.array([4, 2, 0]),
        ]
    )

    # Total length of 6, so halfway along the object
    # would be at length 3, which lands in the first, long line.
    assert np.all(obj.point_from_proportion(0.5) == np.array([3, 0, 0]))

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
    with pytest.raises(Exception):  # TODO change this to ValueError once #307 is merged
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
    with pytest.raises(Exception):  # TODO change this to ValueError once #307 is merged
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
    assert len((obj - a)) == 1
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
