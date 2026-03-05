from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from manim import Typst, TypstMath, tempconfig


def test_Typst(config):
    """Basic Typst creation produces an SVG file."""
    m = Typst(r"$ x^2 $")
    assert m.height > 0
    assert m.width > 0
    assert len(m.submobjects) > 0


def test_TypstMath(config):
    """TypstMath wraps the expression in math delimiters."""
    m = TypstMath(r"alpha + beta")
    assert m.typst_code == "$ alpha + beta $"
    assert m.height > 0


def test_typst_default_font_size(config):
    """Default font_size is 48 (DEFAULT_FONT_SIZE)."""
    m = Typst(r"$ a + b $")
    assert np.isclose(m.font_size, 48)


def test_typst_custom_font_size(config):
    """Passing a custom font_size scales the mobject accordingly."""
    m = Typst(r"$ a + b $", font_size=72)
    assert np.isclose(m.font_size, 72)


def test_typst_font_size_property_setter(config):
    """Setting font_size after creation rescales correctly."""
    m = Typst(r"$ a + b $")
    original_height = m.height
    m.font_size = 96
    assert np.isclose(m.font_size, 96)
    assert m.height > original_height


def test_typst_font_size_error(config):
    """Setting font_size to a non-positive value raises ValueError."""
    m = Typst(r"$ a + b $")
    with pytest.raises(ValueError, match="font_size must be greater than 0"):
        m.font_size = -1


def test_typst_caching(config):
    """Compiling the same source twice uses the cached SVG."""
    m1 = Typst(r"$ e^{i pi} + 1 = 0 $")
    m2 = Typst(r"$ e^{i pi} + 1 = 0 $")
    assert np.isclose(m1.height, m2.height)
    assert np.isclose(m1.width, m2.width)


def test_typst_preamble(config):
    """A custom preamble is accepted without error."""
    m = Typst(
        r"$ x^2 $",
        typst_preamble='#set text(font: "New Computer Modern")',
    )
    assert m.height > 0


def test_typst_repr(config):
    """__repr__ includes the Typst source."""
    m = Typst("hello")
    assert repr(m) == "Typst('hello')"

    m2 = TypstMath("x")
    assert repr(m2) == "TypstMath('$ x $')"


def test_typst_text_rendering(config):
    """Non-math Typst markup renders correctly."""
    m = Typst(r"*Bold* and _italic_")
    assert m.height > 0
    assert len(m.submobjects) > 0
