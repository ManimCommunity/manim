from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

from manim.constants import LEFT, RIGHT
from manim.mobject.text.text_mobject import MarkupText, Paragraph, Text


def test_font_size():
    """Test that Text and MarkupText return the
    correct font_size value after being scaled.
    """
    text_string = Text("0").scale(0.3)
    markuptext_string = MarkupText("0").scale(0.3)

    assert round(text_string.font_size, 5) == 14.4
    assert round(markuptext_string.font_size, 5) == 14.4


def test_font_warnings():
    def warning_printed(font: str, **kwargs) -> bool:
        io = StringIO()
        with redirect_stdout(io):
            Text("hi!", font=font, **kwargs)
        txt = io.getvalue()
        return "Font" in txt and "not in" in txt

    # check for normal fonts (no warning)
    assert not warning_printed("System-ui", warn_missing_font=True)
    # should be converted to sans before checking
    assert not warning_printed("Sans-serif", warn_missing_font=True)

    # check random string (should be warning)
    assert warning_printed("Manim!" * 3, warn_missing_font=True)


def test_paragraph_alignment():
    def float_eq(a: float, b: float, delta: float = 1e-2):
        return abs(a - b) <= delta

    # check paragraph left alignment
    par_left = Paragraph("This is", "a left-aligned", "paragraph", alignment="left")
    assert float_eq(par_left[0][0].get_x(LEFT), par_left[1][0].get_x(LEFT))
    assert float_eq(par_left[0][0].get_x(LEFT), par_left[2][0].get_x(LEFT))

    # check paragraph right alignment
    par_right = Paragraph("This is", "a right-aligned", "paragraph", alignment="right")
    assert float_eq(par_right[0][-1].get_x(RIGHT), par_right[1][-1].get_x(RIGHT))
    assert float_eq(par_right[0][-1].get_x(RIGHT), par_right[2][-1].get_x(RIGHT))

    # check paragraph center alignment
    par_center = Paragraph(
        "This is", "a center-aligned", "paragraph", alignment="center"
    )
    assert float_eq(
        (par_center[0][0].get_x(LEFT) + par_center[0][-1].get_x(RIGHT)) / 2,
        (par_center[1][0].get_x(LEFT) + par_center[1][-1].get_x(RIGHT)) / 2,
    )
    assert float_eq(
        (par_center[0][0].get_x(LEFT) + par_center[0][-1].get_x(RIGHT)) / 2,
        (par_center[2][0].get_x(LEFT) + par_center[2][-1].get_x(RIGHT)) / 2,
    )
