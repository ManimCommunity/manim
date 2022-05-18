from __future__ import annotations

from colour import Color

from manim.mobject.text.text_mobject import MarkupText, Text


def test_font_size():
    """Test that Text and MarkupText return the
    correct font_size value after being scaled."""
    text_string = Text("0").scale(0.3)
    markuptext_string = MarkupText("0").scale(0.3)

    assert round(text_string.font_size, 5) == 14.4
    assert round(markuptext_string.font_size, 5) == 14.4


def test_non_str_color():
    """Test that the Text and MarkupText can accept non_str color values
    i.e. colour.Color(red)."""

    text = Text("test_color_inheritance", color=Color("blue"))
    markup_text = MarkupText("test_color_inheritance", color=Color("blue"))
