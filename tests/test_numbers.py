import pytest

from manim.mobject.numbers import DecimalNumber


def test_font_size():
    """Test that DecimalNumber returns the correct font_size value
    after being scaled."""
    num = DecimalNumber(0).scale(0.3)

    assert round(num.font_size, 5) == 14.4


def test_set_value_size():
    """Test that the size of DecimalNumber after set_value is correct."""
    num = DecimalNumber(0).scale(0.3)
    test_num = num.copy()
    num.set_value(0)

    # round because the height is off by 1e-17
    assert round(num.height, 12) == round(test_num.height, 12)
