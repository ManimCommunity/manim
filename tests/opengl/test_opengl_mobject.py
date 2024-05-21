from __future__ import annotations

import pytest

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
