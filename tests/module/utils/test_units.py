from __future__ import annotations

import numpy as np
import pytest

from manim import DOWN, IN, LEFT, OUT, PI, RIGHT, UL, UP, UR, X_AXIS, Y_AXIS, Z_AXIS
from manim.utils.unit import Degrees, Munits, Percent, Pixels


def test_units(config):
    # make sure we are using the right frame geometry
    config.pixel_width = 1920

    np.testing.assert_allclose(config.frame_height, 8.0)

    # Munits should be equivalent to the internal logical units
    np.testing.assert_allclose(8.0 * Munits, config.frame_height)

    # Pixels should convert from pixels to Munits
    np.testing.assert_allclose(1920 * Pixels, config.frame_width)

    # Percent should give the fractional length of the frame
    np.testing.assert_allclose(50 * Percent(X_AXIS), config.frame_width / 2)
    np.testing.assert_allclose(50 * Percent(Y_AXIS), config.frame_height / 2)

    # The frame has no depth, so the Z axis has length 0
    np.testing.assert_allclose(50 * Percent(Z_AXIS), 0)

    # Degrees should convert from degrees to radians
    np.testing.assert_allclose(180 * Degrees, PI)


@pytest.mark.parametrize("axis", [X_AXIS, RIGHT, LEFT, [-1, 0, 0]])
def test_percent_horizontal_unit_vectors_use_frame_width(config, axis):
    np.testing.assert_allclose(50 * Percent(axis), config.frame_width / 2)


@pytest.mark.parametrize("axis", [Y_AXIS, UP, DOWN])
def test_percent_vertical_unit_vectors_use_frame_height(config, axis):
    np.testing.assert_allclose(50 * Percent(axis), config.frame_height / 2)


@pytest.mark.parametrize("scale", [0.5, 2, 3])
def test_percent_scales_with_vector_length(config, scale):
    np.testing.assert_allclose(
        50 * Percent(scale * LEFT), scale * config.frame_width / 2
    )
    np.testing.assert_allclose(
        50 * Percent(scale * DOWN), scale * config.frame_height / 2
    )


@pytest.mark.parametrize("axis", [UR, UL, [3, 1, 0]])
def test_percent_diagonal_vectors(config, axis):
    expected = np.hypot(axis[0] * config.frame_width, axis[1] * config.frame_height)
    np.testing.assert_allclose(100 * Percent(axis), expected)


def test_percent_diagonal_explicit_value(config):
    # With the default frame (128/9 by 8), UR spans the full frame diagonal.
    np.testing.assert_allclose(100 * Percent(UR), np.hypot(128 / 9, 8))


@pytest.mark.parametrize("axis", [Z_AXIS, OUT, IN, [0, 0, 0]])
def test_percent_zero_depth_and_zero_vector_give_zero(axis):
    np.testing.assert_allclose(50 * Percent(axis), 0)


def test_percent_z_component_does_not_contribute(config):
    np.testing.assert_allclose(50 * Percent(RIGHT + OUT), config.frame_width / 2)


@pytest.mark.parametrize("axis", [[np.nan, 0, 0], [np.inf, 0, 0], [1, np.nan, 0]])
def test_percent_rejects_non_finite_vectors(axis):
    with pytest.raises(ValueError):
        Percent(axis)


def test_percent_follows_config_changes(config):
    percent = Percent(RIGHT)
    config.frame_width = 20
    np.testing.assert_allclose(50 * percent, 10)
