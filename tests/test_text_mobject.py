import pytest

from manim.mobject.svg.text_mobject import MarkupText, Text


def test_font_size():
    """Test that Text and MarkupText return the
    correct font_size value after being scaled."""
    text_string = Text("0").scale(0.3)
    markuptext_string = MarkupText("0").scale(0.3)

    assert round(text_string.font_size, 5) == 14.4
    assert round(markuptext_string.font_size, 5) == 14.4
