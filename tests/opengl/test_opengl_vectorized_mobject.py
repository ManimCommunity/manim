from __future__ import annotations

import numpy as np
import pytest

from manim import Circle, Line, Square, VDict, VGroup, VMobject
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject


def test_opengl_vmobject_add(using_opengl_renderer):
    """Test the OpenGLVMobject add method."""
    obj = OpenGLVMobject()
    assert len(obj.submobjects) == 0

    obj.add(OpenGLVMobject())
    assert len(obj.submobjects) == 1

    # Can't add non-OpenGLVMobject values to a VMobject.
    with pytest.raises(TypeError) as add_int_info:
        obj.add(3)
    assert str(add_int_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "OpenGLVMobject, but the value 3 (at index 0) is of type int."
    )
    assert len(obj.submobjects) == 1

    # Plain OpenGLMobjects can't be added to a OpenGLVMobject if they're not
    # OpenGLVMobjects. Suggest adding them into an OpenGLGroup instead.
    with pytest.raises(TypeError) as add_mob_info:
        obj.add(OpenGLMobject())
    assert str(add_mob_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "OpenGLVMobject, but the value OpenGLMobject (at index 0) is of type "
        "OpenGLMobject. You can try adding this value into a Group instead."
    )
    assert len(obj.submobjects) == 1

    with pytest.raises(TypeError) as add_vmob_and_mob_info:
        # If only one of the added objects is not an instance of VMobject, none of them should be added
        obj.add(OpenGLVMobject(), OpenGLMobject())
    assert str(add_vmob_and_mob_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "OpenGLVMobject, but the value OpenGLMobject (at index 1) is of type "
        "OpenGLMobject. You can try adding this value into a Group instead."
    )
    assert len(obj.submobjects) == 1

    # A VMobject or VGroup cannot contain itself.
    with pytest.raises(ValueError) as add_self_info:
        obj.add(obj)
    assert str(add_self_info.value) == (
        "Cannot add OpenGLVMobject as a submobject of itself (at index 0)."
    )
    assert len(obj.submobjects) == 1


def test_opengl_vmobject_point_from_proportion(using_opengl_renderer):
    obj = OpenGLVMobject()

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


def test_vgroup_init(using_opengl_renderer):
    """Test the VGroup instantiation."""
    VGroup()
    VGroup(OpenGLVMobject())
    VGroup(OpenGLVMobject(), OpenGLVMobject())

    # A VGroup cannot contain non-VMobject values.
    with pytest.raises(TypeError) as init_with_float_info:
        VGroup(3.0)
    assert str(init_with_float_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "VGroup, but the value 3.0 (at index 0 of parameter 0) is of type float."
    )

    with pytest.raises(TypeError) as init_with_mob_info:
        VGroup(OpenGLMobject())
    assert str(init_with_mob_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "VGroup, but the value OpenGLMobject (at index 0 of parameter 0) is of type "
        "OpenGLMobject. You can try adding this value into a Group instead."
    )

    with pytest.raises(TypeError) as init_with_vmob_and_mob_info:
        VGroup(OpenGLVMobject(), OpenGLMobject())
    assert str(init_with_vmob_and_mob_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "VGroup, but the value OpenGLMobject (at index 0 of parameter 1) is of type "
        "OpenGLMobject. You can try adding this value into a Group instead."
    )


def test_vgroup_init_with_iterable(using_opengl_renderer):
    """Test VGroup instantiation with an iterable type."""

    def type_generator(type_to_generate, n):
        return (type_to_generate() for _ in range(n))

    def mixed_type_generator(major_type, minor_type, minor_type_positions, n):
        return (
            minor_type() if i in minor_type_positions else major_type()
            for i in range(n)
        )

    obj = VGroup(OpenGLVMobject())
    assert len(obj.submobjects) == 1

    obj = VGroup(type_generator(OpenGLVMobject, 38))
    assert len(obj.submobjects) == 38

    obj = VGroup(
        OpenGLVMobject(),
        [OpenGLVMobject(), OpenGLVMobject()],
        type_generator(OpenGLVMobject, 38),
    )
    assert len(obj.submobjects) == 41

    # A VGroup cannot be initialised with an iterable containing a OpenGLMobject
    with pytest.raises(TypeError) as init_with_mob_iterable:
        VGroup(type_generator(OpenGLMobject, 5))
    assert str(init_with_mob_iterable.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of VGroup, "
        "but the value OpenGLMobject (at index 0 of parameter 0) is of type OpenGLMobject."
    )

    # A VGroup cannot be initialised with an iterable containing a OpenGLMobject in any position
    with pytest.raises(TypeError) as init_with_mobs_and_vmobs_iterable:
        VGroup(mixed_type_generator(OpenGLVMobject, OpenGLMobject, [3, 5], 7))
    assert str(init_with_mobs_and_vmobs_iterable.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of VGroup, "
        "but the value OpenGLMobject (at index 3 of parameter 0) is of type OpenGLMobject."
    )

    # A VGroup cannot be initialised with an iterable containing non OpenGLVMobject's in any position
    with pytest.raises(TypeError) as init_with_float_and_vmobs_iterable:
        VGroup(mixed_type_generator(OpenGLVMobject, float, [6, 7], 9))
    assert str(init_with_float_and_vmobs_iterable.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of VGroup, "
        "but the value 0.0 (at index 6 of parameter 0) is of type float."
    )

    # A VGroup cannot be initialised with an iterable containing both OpenGLVMobject's and VMobject's
    with pytest.raises(TypeError) as init_with_mobs_and_vmobs_iterable:
        VGroup(mixed_type_generator(OpenGLVMobject, VMobject, [3, 5], 7))
    assert str(init_with_mobs_and_vmobs_iterable.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of VGroup, "
        "but the value VMobject (at index 3 of parameter 0) is of type VMobject."
    )


def test_vgroup_add(using_opengl_renderer):
    """Test the VGroup add method."""
    obj = VGroup()
    assert len(obj.submobjects) == 0

    obj.add(OpenGLVMobject())
    assert len(obj.submobjects) == 1

    # Can't add non-OpenGLVMobject values to a VMobject.
    with pytest.raises(TypeError) as add_int_info:
        obj.add(3)
    assert str(add_int_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "VGroup, but the value 3 (at index 0 of parameter 0) is of type int."
    )
    assert len(obj.submobjects) == 1

    # Plain OpenGLMobjects can't be added to a OpenGLVMobject if they're not
    # OpenGLVMobjects. Suggest adding them into an OpenGLGroup instead.
    with pytest.raises(TypeError) as add_mob_info:
        obj.add(OpenGLMobject())
    assert str(add_mob_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "VGroup, but the value OpenGLMobject (at index 0 of parameter 0) is of type "
        "OpenGLMobject. You can try adding this value into a Group instead."
    )
    assert len(obj.submobjects) == 1

    with pytest.raises(TypeError) as add_vmob_and_mob_info:
        # If only one of the added objects is not an instance of VMobject, none of them should be added
        obj.add(OpenGLVMobject(), OpenGLMobject())
    assert str(add_vmob_and_mob_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "VGroup, but the value OpenGLMobject (at index 0 of parameter 1) is of type "
        "OpenGLMobject. You can try adding this value into a Group instead."
    )
    assert len(obj.submobjects) == 1

    # A VMobject or VGroup cannot contain itself.
    with pytest.raises(ValueError) as add_self_info:
        obj.add(obj)
    assert str(add_self_info.value) == (
        "Cannot add VGroup as a submobject of itself (at index 0)."
    )
    assert len(obj.submobjects) == 1


def test_vgroup_add_dunder(using_opengl_renderer):
    """Test the VGroup __add__ magic method."""
    obj = VGroup()
    assert len(obj.submobjects) == 0
    obj + OpenGLVMobject()
    assert len(obj.submobjects) == 0
    obj += OpenGLVMobject()
    assert len(obj.submobjects) == 1
    with pytest.raises(TypeError):
        obj += OpenGLMobject()
    assert len(obj.submobjects) == 1
    with pytest.raises(TypeError):
        # If only one of the added object is not an instance of OpenGLVMobject, none of them should be added
        obj += (OpenGLVMobject(), OpenGLMobject())
    assert len(obj.submobjects) == 1
    with pytest.raises(ValueError):
        # a OpenGLMobject cannot contain itself
        obj += obj


def test_vgroup_remove(using_opengl_renderer):
    """Test the VGroup remove method."""
    a = OpenGLVMobject()
    c = OpenGLVMobject()
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


def test_vgroup_remove_dunder(using_opengl_renderer):
    """Test the VGroup __sub__ magic method."""
    a = OpenGLVMobject()
    c = OpenGLVMobject()
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


def test_vmob_add_to_back(using_opengl_renderer):
    """Test the OpenGLMobject add_to_back method."""
    a = OpenGLVMobject()
    b = Line()
    c = "text"
    with pytest.raises(ValueError):
        # OpenGLMobject cannot contain self
        a.add_to_back(a)
    with pytest.raises(TypeError):
        # All submobjects must be of type OpenGLMobject
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


def test_vdict_init(using_opengl_renderer):
    """Test the VDict instantiation."""
    # Test empty VDict
    VDict()
    # Test VDict made from list of pairs
    VDict([("a", OpenGLVMobject()), ("b", OpenGLVMobject()), ("c", OpenGLVMobject())])
    # Test VDict made from a python dict
    VDict({"a": OpenGLVMobject(), "b": OpenGLVMobject(), "c": OpenGLVMobject()})
    # Test VDict made using zip
    VDict(
        zip(
            ["a", "b", "c"],
            [OpenGLVMobject(), OpenGLVMobject(), OpenGLVMobject()],
            strict=False,
        )
    )
    # If the value is of type OpenGLMobject, must raise a TypeError
    with pytest.raises(TypeError):
        VDict({"a": OpenGLMobject()})


def test_vdict_add(using_opengl_renderer):
    """Test the VDict add method."""
    obj = VDict()
    assert len(obj.submob_dict) == 0
    obj.add([("a", OpenGLVMobject())])
    assert len(obj.submob_dict) == 1
    with pytest.raises(TypeError):
        obj.add([("b", OpenGLMobject())])


def test_vdict_remove(using_opengl_renderer):
    """Test the VDict remove method."""
    obj = VDict([("a", OpenGLVMobject())])
    assert len(obj.submob_dict) == 1
    obj.remove("a")
    assert len(obj.submob_dict) == 0
    with pytest.raises(KeyError):
        obj.remove("a")


def test_vgroup_supports_item_assigment(using_opengl_renderer):
    """Test VGroup supports array-like assignment for OpenGLVMObjects"""
    a = OpenGLVMobject()
    b = OpenGLVMobject()
    vgroup = VGroup(a)
    assert vgroup[0] == a
    vgroup[0] = b
    assert vgroup[0] == b
    assert len(vgroup) == 1


def test_vgroup_item_assignment_at_correct_position(using_opengl_renderer):
    """Test VGroup item-assignment adds to correct position for OpenGLVMobjects"""
    n_items = 10
    vgroup = VGroup()
    for _i in range(n_items):
        vgroup.add(OpenGLVMobject())
    new_obj = OpenGLVMobject()
    vgroup[6] = new_obj
    assert vgroup[6] == new_obj
    assert len(vgroup) == n_items


def test_vgroup_item_assignment_only_allows_vmobjects(using_opengl_renderer):
    """Test VGroup item-assignment raises TypeError when invalid type is passed"""
    vgroup = VGroup(OpenGLVMobject())
    with pytest.raises(TypeError) as assign_str_info:
        vgroup[0] = "invalid object"
    assert str(assign_str_info.value) == (
        "Only values of type OpenGLVMobject can be added as submobjects of "
        "VGroup, but the value invalid object (at index 0) is of type str."
    )
