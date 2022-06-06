from __future__ import annotations

import numpy as np
import numpy.testing as nt
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
    obj = OpenGLMobject()

    # a OpenGLMobject cannot contain itself
    with pytest.raises(ValueError):
        obj.add(obj)

    # can only add OpenGLMobjects
    with pytest.raises(TypeError):
        obj.add("foo")


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


def test_opengl_mobject_arrange_in_grid(using_opengl_renderer):
    """Test OpenGLMobject.arrange_in_grid()."""
    from manim import Rectangle, VGroup

    boxes = VGroup(*[Rectangle(height=0.5, width=0.5) for _ in range(24)])

    boxes.arrange_in_grid(
        buff=(0.25, 0.5),
        col_alignments="lccccr",
        row_alignments="uccd",
        col_widths=[1, *[None] * 4, 1],
        row_heights=[1, None, None, 1],
        flow_order="dr",
    )

    solution = np.array(
        [
            [-2.375, 2.0, 0.0],
            [-2.375, 0.5, 0.0],
            [-2.375, -0.5, 0.0],
            [-2.375, -2.0, 0.0],
            [-1.125, 2.0, 0.0],
            [-1.125, 0.5, 0.0],
            [-1.125, -0.5, 0.0],
            [-1.125, -2.0, 0.0],
            [-0.375, 2.0, 0.0],
            [-0.375, 0.5, 0.0],
            [-0.375, -0.5, 0.0],
            [-0.375, -2.0, 0.0],
            [0.375, 2.0, 0.0],
            [0.375, 0.5, 0.0],
            [0.375, -0.5, 0.0],
            [0.375, -2.0, 0.0],
            [1.125, 2.0, 0.0],
            [1.125, 0.5, 0.0],
            [1.125, -0.5, 0.0],
            [1.125, -2.0, 0.0],
            [2.375, 2.0, 0.0],
            [2.375, 0.5, 0.0],
            [2.375, -0.5, 0.0],
            [2.375, -2.0, 0.0],
        ]
    )

    for expected, rect in zip(solution, boxes):
        c = rect.get_center()
        nt.assert_array_equal(expected, c)


# for rect in boxes:
#     c = rect.get_center()
#     print(f"{c!s}")
