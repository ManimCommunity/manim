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
    assert len(obj.submobjects) == 0
    obj.add(OpenGLMobject())
    assert len(obj.submobjects) == 1
    obj.add(*(OpenGLMobject() for _ in range(10)))
    assert len(obj.submobjects) == 11

    # check that adding a OpenGLMobject twice does not actually add it twice
    repeated = OpenGLMobject()
    obj.add(repeated)
    assert len(obj.submobjects) == 12
    obj.add(repeated)
    assert len(obj.submobjects) == 12

    # check that OpenGLMobject.add() returns the OpenGLMobject (for chained calls)
    assert obj.add(OpenGLMobject()) is obj
    assert len(obj.submobjects) == 13

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


def test_opengl_mobject_family_updated_on_change(using_opengl_renderer):
    """Test that the family of an OpenGLMobject is updated correctly when submobjects
    are added or removed, and that the family is not updated if no changes are made.
    """
    # This is based on the assumption that obj.family is replaced with a new list when submobjects
    # are added or removed. If this implementation detail changes, this test may need to be updated.
    obj = OpenGLMobject()
    family = obj.get_family()
    assert family == [obj]
    submobs = [OpenGLMobject() for _ in range(10)]

    # Add new submobjects; family should be updated.
    obj.add(*submobs)
    assert family is not obj.get_family()
    family = obj.get_family()
    assert len(family) == 11
    for submob in submobs:
        assert submob in family

    # Remove a submobject; family should be updated.
    obj.remove(submobs[0])
    family = obj.get_family()
    assert len(family) == 10
    assert submobs[0] not in family

    # Remove a submobject that is not in the family; family should not be updated.
    obj.remove(OpenGLMobject())
    assert family is obj.get_family()

    # Add a submobject that is already in the family; family should not be updated.
    obj.add(submobs[1])
    assert family is obj.get_family()

    # Add a mix of new and existing submobjects; family should be updated.
    obj.add(OpenGLMobject(), submobs[2])
    assert family is not obj.get_family()
    family = obj.get_family()
    assert len(family) == 11

    # Remove a mix of existing and non-existing submobjects; family should be updated.
    obj.remove(submobs[3], OpenGLMobject())
    assert family is not obj.get_family()


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
