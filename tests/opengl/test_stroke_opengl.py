from __future__ import annotations

import manim.utils.color as C
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject


def test_stroke_props_in_ctor(using_opengl_renderer):
    m = OpenGLVMobject(stroke_color=C.ORANGE, stroke_width=10)
    assert m.stroke_color.to_hex() == C.ORANGE.to_hex()
    assert m.stroke_width == 10


def test_set_stroke(using_opengl_renderer):
    m = OpenGLVMobject()
    m.set_stroke(color=C.ORANGE, width=2, opacity=0.8)
    assert m.stroke_width == 2
    assert m.stroke_opacity == 0.8
    assert m.stroke_color.to_hex() == C.ORANGE.to_hex()
