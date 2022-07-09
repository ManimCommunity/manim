from __future__ import annotations

from colour import Color

import manim.utils.color as C
from manim import VMobject


def test_stroke_props_in_ctor():
    m = VMobject(stroke_color=C.ORANGE, stroke_width=10)
    assert m.stroke_color == Color(C.ORANGE)
    assert m.stroke_width == 10


def test_set_stroke():
    m = VMobject()
    m.set_stroke(color=C.ORANGE, width=2, opacity=0.8)
    assert m.stroke_width == 2
    assert m.stroke_opacity == 0.8
    assert m.stroke_color == Color(C.ORANGE)


def test_set_background_stroke():
    m = VMobject()
    m.set_stroke(color=C.ORANGE, width=2, opacity=0.8, background=True)
    assert m.background_stroke_width == 2
    assert m.background_stroke_opacity == 0.8
    assert m.background_stroke_color == C.ORANGE
