from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import pytest

from manim.mobject.text.text_mobject import MarkupText, Text


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


def test_gen_chars_raises_clear_error_on_glyph_mismatch():
    """``_gen_chars`` should raise a clear, actionable error instead of an
    opaque ``IndexError`` when the number of rendered glyphs doesn't match
    the number of non-space characters. This happens when a font implements
    some of its ligatures (e.g. programming ligatures like ``<=``) through
    an OpenType feature that ``disable_ligatures`` doesn't disable, such as
    ``calt`` (see issue #3237). This test simulates that mismatch directly,
    without depending on any particular font being installed.
    """
    text = Text("ab", disable_ligatures=True)
    # Simulate a font that merged "ab" into a single ligature glyph.
    text.submobjects = text.submobjects[:1]
    with pytest.raises(ValueError, match="glyph"):
        text._gen_chars()
