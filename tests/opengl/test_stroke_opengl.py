from __future__ import annotations

import numpy.testing as nt
from colour import Color

import manim.utils.color as C
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject


def test_stroke_props_in_ctor(using_opengl_renderer):
    m = OpenGLVMobject(stroke_color=C.ORANGE, stroke_width=10)
    assert m.stroke_color == C.ORANGE
    assert m.stroke_width == 10


def test_set_stroke(using_opengl_renderer):
    m = OpenGLVMobject()
    m.set_stroke(color=C.ORANGE, width=2, opacity=0.8)
    assert m.stroke_width == 2
    assert m.stroke_opacity == 0.8
    assert m.stroke_color == C.ORANGE


# def test_set_stroke_list(using_opengl_renderer):
#     m = OpenGLVMobject()
#     m.set_stroke([C.RED, C.ORANGE], [1, 2, 3], [0, 0.5, 1])
#     assert m.stroke_width == 1
#     assert m.stroke_opacity == 0
#     assert m.stroke_color == C.RED
#     m.stroke_opacity = [0.1, 0.2, 0.3]
#     assert m.stroke_opacity == 0.1
#     nt.assert_array_equal(m.get_stroke_opacities(), [0.1, 0.2, 0.3])
