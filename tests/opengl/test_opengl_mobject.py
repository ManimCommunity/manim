from __future__ import annotations

import numpy as np
import pytest

from manim import PI
from manim.mobject.opengl.opengl_geometry import OpenGLTriangle
from manim.mobject.opengl.opengl_mobject import OpenGLMobject


def test_opengl_mobject_add(using_opengl_renderer):
    """Test OpenGLMobject.add()."""
    """Call this function with a Container instance to test its add() method."""
    # check that obj.submobjects is updated correctly
    obj = OpenGLMobject()
    assert obj.submobjects == []
    child = OpenGLMobject()
    obj.add(child)
    assert obj.submobjects == [child]
    new_mobjects = [OpenGLMobject() for _ in range(10)]
    obj.add(*new_mobjects)
    assert obj.submobjects == [child] + new_mobjects

    # check that adding a OpenGLMobject twice does not actually add it twice
    repeated = OpenGLMobject()
    obj.add(repeated)
    assert len(obj.submobjects) == 12
    obj.add(repeated)
    assert len(obj.submobjects) == 12

    # check that adding already existing mobjects doesn't affect order
    obj.add(*reversed(new_mobjects))
    assert obj.submobjects == [child] + new_mobjects + [repeated]

    # check that OpenGLMobject.add() returns the OpenGLMobject (for chained calls)
    assert obj.add(OpenGLMobject()) is obj
    assert len(obj.submobjects) == 13

    obj = OpenGLMobject()
    # check that inserting duplicate mobjects keeps the first occurrence
    m1, m2 = OpenGLMobject(), OpenGLMobject()
    obj.add(m1, m2, m1)
    assert obj.submobjects == [m1, m2]

    obj = OpenGLMobject()

    # an OpenGLMobject cannot contain itself
    with pytest.raises(ValueError) as add_self_info:
        obj.add(OpenGLMobject(), obj, OpenGLMobject())
    assert str(add_self_info.value) == (
        "Cannot add OpenGLMobject as a submobject of itself (at index 1)."
    )
    assert len(obj.submobjects) == 0

    # can only add Mobjects
    with pytest.raises(TypeError) as add_str_info:
        obj.add(OpenGLMobject(), OpenGLMobject(), "foo")
    assert str(add_str_info.value) == (
        "Only values of type OpenGLMobject can be added as submobjects of "
        "OpenGLMobject, but the value foo (at index 2) is of type str."
    )
    assert len(obj.submobjects) == 0


def test_opengl_mobject_insert(using_opengl_renderer):
    obj = OpenGLMobject()
    m1, m2, m3, m4 = [OpenGLMobject(name=f"m{i}") for i in range(1, 5)]
    # Insert into empty list
    obj.insert(0, m1)
    assert obj.submobjects == [m1]

    # Inserting shifts existing submobjects to the right
    obj.insert(0, m2)
    assert obj.submobjects == [m2, m1]

    # Inserting with negative index inserts counting from the end like a list
    obj.insert(-1, m3)
    assert obj.submobjects == [m2, m3, m1]

    # Inserting with index greater than length appends
    obj.insert(10, m4)
    assert obj.submobjects == [m2, m3, m1, m4]

    # Inserting an existing submobject does nothing
    for i in range(len(obj.submobjects) + 1):
        obj.insert(i, m3)
        assert obj.submobjects == [m2, m3, m1, m4]

    # obj cannot insert itself
    with pytest.raises(ValueError) as insert_self_info:
        obj.insert(0, obj)
    assert str(insert_self_info.value) == (
        "Cannot add OpenGLMobject as a submobject of itself (at index 0)."
    )

    # can only add Mobjects
    with pytest.raises(TypeError) as insert_str_info:
        obj.insert(0, "foo")
    assert str(insert_str_info.value) == (
        "Only values of type OpenGLMobject can be added as submobjects of "
        "OpenGLMobject, but the value foo (at index 0) is of type str."
    )


def test_opengl_mobject_add_to_back(using_opengl_renderer):
    obj = OpenGLMobject()
    m1, m2, m3, m4 = [OpenGLMobject(name=f"m{i}") for i in range(1, 5)]

    # Adding to empty list is the same as adding normally
    obj.add_to_back(m1)
    assert obj.submobjects == [m1]

    # Adding a new submobject adds it to the back
    obj.add_to_back(m2)
    assert obj.submobjects == [m2, m1]

    # In case of duplicate input, the first occurrence of a submobject is kept
    obj.add_to_back(m3, m4, m3)
    assert obj.submobjects == [m3, m4, m2, m1]

    # Adding an existing submobject does nothing
    for mob in obj.submobjects:
        obj.add_to_back(mob)
        assert obj.submobjects == [m3, m4, m2, m1]

    # can only add Mobjects
    with pytest.raises(TypeError) as add_str_info:
        obj.add_to_back("foo")
    assert str(add_str_info.value) == (
        "Only values of type OpenGLMobject can be added as submobjects of "
        "OpenGLMobject, but the value foo (at index 0) is of type str."
    )


def test_opengl_mobject_remove(using_opengl_renderer):
    """Test OpenGLMobject.remove()."""
    obj = OpenGLMobject()
    to_remove = OpenGLMobject()
    obj.add(to_remove)
    obj.add(*(OpenGLMobject() for _ in range(10)))
    assert len(obj.submobjects) == 11
    obj.remove(to_remove)
    assert len(obj.submobjects) == 10
    obj.remove(to_remove)
    assert len(obj.submobjects) == 10

    assert obj.remove(OpenGLMobject()) is obj


def test_opengl_mobject_get_boundary_point(using_opengl_renderer):
    """Test that get_boundary_point returns the furthest point in a direction."""
    obj = OpenGLMobject().set_points(
        np.array([[-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    np.testing.assert_array_equal(obj.get_boundary_point([1, 0, 0]), [2, 0, 0])


def test_opengl_mobject_stretch_to_fit_depth(using_opengl_renderer):
    """Test that stretch_to_fit_depth changes the depth dimension."""
    obj = OpenGLMobject().set_points(
        np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]]),
    )

    obj.stretch_to_fit_depth(12)

    np.testing.assert_allclose([obj.width, obj.height, obj.depth], [2, 4, 12])


def test_opengl_rotate_about_vertex_view(using_opengl_renderer):
    """Test that rotating about a vertex obtained from get_vertices() works correctly.

    This is a regression test for an issue in the non-OpenGL (Cairo) renderer where
    get_vertices() returns a view of the points array, and using it as about_point
    in rotate() would cause the view to be mutated. The OpenGL renderer was not affected
    by this bug due to its different implementation (using `arr - about_point` which
    creates a temporary array rather than `arr -= about_point` which mutates in-place).

    This test verifies that the OpenGL renderer continues to handle vertex views correctly.
    """
    triangle = OpenGLTriangle()
    original_vertices = triangle.get_vertices().copy()
    first_vertex = original_vertices[0].copy()

    # This should rotate about the first vertex without corrupting it
    triangle.rotate(PI / 2, about_point=triangle.get_vertices()[0])

    # The first vertex should remain in the same position (within numerical precision)
    rotated_vertices = triangle.get_vertices()
    np.testing.assert_allclose(rotated_vertices[0], first_vertex, atol=1e-6)


def test_replace_submobject(using_opengl_renderer):
    """Test that replace_submobject() puts the new submobject in the correct
    place and removes the old one.
    """
    parent = OpenGLMobject()
    old_submobs = [OpenGLMobject() for _ in range(3)]
    parent.add(*old_submobs)
    new_submob = OpenGLMobject()

    parent.replace_submobject(1, new_submob)

    assert parent.submobjects == [old_submobs[0], new_submob, old_submobs[2]]
    assert old_submobs[1] not in parent.submobjects


def test_replace_submobject_with_existing_submobject(using_opengl_renderer):
    """Test that replacing with a submobject that is already present moves it
    to the new index instead of duplicating it.
    """
    parent = OpenGLMobject()
    submobs = [OpenGLMobject() for _ in range(3)]
    parent.add(*submobs)

    parent.replace_submobject(0, submobs[2])

    assert parent.submobjects == [submobs[2], submobs[1]]
