from __future__ import annotations

from colour import Color

from manim import RED, DecimalNumber, Integer


def test_font_size():
    """Test that DecimalNumber returns the correct font_size value
    after being scaled."""
    num = DecimalNumber(0).scale(0.3)

    assert round(num.font_size, 5) == 14.4


def test_font_size_vs_scale():
    """Test that scale produces the same results as .scale()"""
    num = DecimalNumber(0, font_size=12)
    num_scale = DecimalNumber(0).scale(1 / 4)

    assert num.height == num_scale.height


def test_changing_font_size():
    """Test that the font_size property properly scales DecimalNumber."""
    num = DecimalNumber(0, font_size=12)
    num.font_size = 48

    assert num.height == DecimalNumber(0, font_size=48).height


def test_set_value_size():
    """Test that the size of DecimalNumber after set_value is correct."""
    num = DecimalNumber(0).scale(0.3)
    test_num = num.copy()
    num.set_value(0)

    # round because the height is off by 1e-17
    assert round(num.height, 12) == round(test_num.height, 12)


def test_color_when_number_of_digits_changes():
    """Test that all digits of an Integer are colored correctly when
    the number of digits changes."""
    mob = Integer(color=RED)
    mob.set_value(42)
    assert all([submob.stroke_color == Color(RED) for submob in mob.submobjects])
