from __future__ import annotations

import numpy as np
import pytest

from manim import (
    DEGREES,
    LEFT,
    ORIGIN,
    RED,
    UP,
    ChangeDecimalToValue,
    DecimalNumber,
    Integer,
)


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


def _transformed_layout(value, start_value, transform, **kwargs):
    """Return how ``DecimalNumber(start_value, **kwargs)`` should look after
    ``transform`` and then ``set_value(value)``.
    """
    expected = DecimalNumber(value, **kwargs)
    expected.move_to(DecimalNumber(start_value, **kwargs).get_left(), LEFT)
    return transform(expected)


@pytest.mark.parametrize(
    "transform",
    [
        lambda m: m.rotate(45 * DEGREES, about_point=ORIGIN),
        lambda m: (
            m.scale(0.5, about_point=ORIGIN)
            .rotate(-30 * DEGREES, about_point=ORIGIN)
            .shift(UP)
        ),
        lambda m: m.stretch(2, dim=0, about_point=ORIGIN),
        lambda m: m.rotate(60 * DEGREES, axis=UP, about_point=ORIGIN),
    ],
    ids=["rotate", "scale_rotate_shift", "stretch", "rotate_3d"],
)
def test_set_value_keeps_transformation(transform):
    """Test that set_value keeps the rotation and size of a transformed
    DecimalNumber instead of resetting the rotation and matching the
    height of its bounding box.
    """
    num = transform(DecimalNumber(0, font_size=96))
    num.set_value(12.5)

    expected = _transformed_layout(12.5, 0, transform, font_size=96)
    np.testing.assert_allclose(
        num.get_all_points(), expected.get_all_points(), atol=1e-8
    )


def test_set_value_while_rotating():
    """Test that a DecimalNumber rotated on every frame of a
    ChangeDecimalToValue keeps its size.
    """
    num = DecimalNumber(0.1234, num_decimal_places=4)
    animation = ChangeDecimalToValue(num, 0.5678)
    animation.begin()
    for alpha in np.linspace(0, 1, 21)[1:]:
        num.rotate(0.05, about_point=ORIGIN)
        animation.interpolate(alpha)

    expected = _transformed_layout(
        0.5678,
        0.1234,
        lambda m: m.rotate(20 * 0.05, about_point=ORIGIN),
        num_decimal_places=4,
    )
    np.testing.assert_allclose(
        num.get_all_points(), expected.get_all_points(), atol=1e-8
    )


def test_set_value_after_changing_points():
    """Test that set_value still matches the size of the number when its
    points no longer correspond to the digits it was built from.
    """
    num = DecimalNumber(0).scale(2)
    num.family_members_with_points()[0].insert_n_curves(3)
    num.set_value(1)

    assert np.isclose(num.height, DecimalNumber(1).scale(2).height)


def test_color_when_number_of_digits_changes():
    """Test that all digits of an Integer are colored correctly when
    the number of digits changes.
    """
    mob = Integer(color=RED)
    mob.set_value(42)
    assert all(
        submob.stroke_color.to_hex() == RED.to_hex() for submob in mob.submobjects
    )
