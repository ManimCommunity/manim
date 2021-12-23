import pytest

from manim.mobject.svg.text_mobject import MarkupText, Paragraph, Text


def test_font_size():
    """Test that Text and MarkupText return the
    correct font_size value after being scaled."""
    text_string = Text("0").scale(0.3)
    markuptext_string = MarkupText("0").scale(0.3)

    assert round(text_string.font_size, 5) == 14.4
    assert round(markuptext_string.font_size, 5) == 14.4


def test_paragraph_singleline():
    """Test a Paragraph with a single line."""
    # need `__eq__` for `Text` to make some tests more useful
    # going to hold off for the `Text` api rework
    # e.g. `assert par[0] == Text("first")`

    # disable ligatures until we decide on a common font for all platforms
    par = Paragraph("first", disable_ligatures=True)

    # flatten
    assert par.lines_text.text == "first"

    # lines
    assert par[0].text == "first"

    # alignment
    assert par.lines[1] == [None]

    # shape
    assert len(par) == 1
    assert list(map(len, par)) == [5]


def test_paragraph_multiline():
    """Test a Paragraph with multiple lines."""
    # see note under `test_paragraph_singleline` about better tests

    # disable ligatures until we decide on a common font for all platforms
    par = Paragraph("first", "second", "third", disable_ligatures=True)

    # flatten
    assert par.lines_text.text == "firstsecondthird"

    # lines
    assert par[0].text == "first"
    assert par[1].text == "second"
    assert par[2].text == "third"

    # alignment
    assert par.lines[1] == [None, None, None]

    # shape
    assert len(par) == 3
    assert list(map(len, par)) == [5, 6, 5]
