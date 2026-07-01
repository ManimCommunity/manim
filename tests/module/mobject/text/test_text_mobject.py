from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

from manim.mobject.text.text_mobject import MarkupText, Text


def test_font_size():
    """Test that Text and MarkupText return the
    correct font_size value after being scaled.
    """
    text_string = Text("0").scale(0.3)
    markuptext_string = MarkupText("0").scale(0.3)

    assert round(text_string.font_size, 5) == 14.4
    assert round(markuptext_string.font_size, 5) == 14.4


def test_z_index_propagates_to_glyphs():
    """A z_index passed to the constructor must reach the glyph submobjects,
    otherwise the z-ordering is ignored during rendering (see issue #4667).
    """
    text = Text("AB", z_index=3)
    assert text.z_index == 3
    assert all(glyph.z_index == 3 for glyph in text.family_members_with_points())

    markup = MarkupText("AB", z_index=3)
    assert all(glyph.z_index == 3 for glyph in markup.family_members_with_points())

    # The default must remain unchanged.
    assert all(glyph.z_index == 0 for glyph in Text("AB").family_members_with_points())


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
