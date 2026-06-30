from __future__ import annotations

from manim import DEGREES, RED, RIGHT, DecimalNumber, Integer
from manim.utils.space_ops import angle_between_vectors


def test_font_size():
    """Test that DecimalNumber returns the correct font_size value
    after being scaled.
    """
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


def test_set_value_preserves_rotation():
    """Test that set_value keeps rotation applied to the number."""
    num = DecimalNumber(0, font_size=48)
    num.rotate(45 * DEGREES)
    direction = num.get_right() - num.get_center()
    angle_before = angle_between_vectors(RIGHT, direction)

    num.set_value(1)

    direction = num.get_right() - num.get_center()
    angle_after = angle_between_vectors(RIGHT, direction)
    assert angle_before == angle_after


def test_set_value_preserves_font_size_when_rotated():
    num = DecimalNumber(0, font_size=48)
    num.rotate(45 * DEGREES)
    num.set_value(1)
    assert num._font_size == 48


def test_color_when_number_of_digits_changes():
    """Test that all digits of an Integer are colored correctly when
    the number of digits changes.
    """
    mob = Integer(color=RED)
    mob.set_value(42)
    assert all(
        submob.stroke_color.to_hex() == RED.to_hex() for submob in mob.submobjects
    )
