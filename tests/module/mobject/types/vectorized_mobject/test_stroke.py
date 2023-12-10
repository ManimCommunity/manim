from __future__ import annotations

import manim.utils.color as C
from manim import VMobject
from manim.mobject.vector_field import StreamLines


def test_stroke_props_in_ctor():
    m = VMobject(stroke_color=C.ORANGE, stroke_width=10)
    assert m.stroke_color.to_hex() == C.ORANGE.to_hex()
    assert m.stroke_width == 10


def test_set_stroke():
    m = VMobject()
    m.set_stroke(color=C.ORANGE, width=2, opacity=0.8)
    assert m.stroke_width == 2
    assert m.stroke_opacity == 0.8
    assert m.stroke_color.to_hex() == C.ORANGE.to_hex()


def test_set_background_stroke():
    m = VMobject()
    m.set_stroke(color=C.ORANGE, width=2, opacity=0.8, background=True)
    assert m.background_stroke_width == 2
    assert m.background_stroke_opacity == 0.8
    assert m.background_stroke_color.to_hex() == C.ORANGE.to_hex()


def test_streamline_attributes_for_single_color():
    vector_field = StreamLines(
        lambda x: x,  # It is not important what this function is.
        x_range=[-1, 1, 0.1],
        y_range=[-1, 1, 0.1],
        padding=0.1,
        stroke_width=1.0,
        opacity=0.2,
        color=C.BLUE_D,
    )
    assert vector_field[0].stroke_width == 1.0
    assert vector_field[0].stroke_opacity == 0.2
