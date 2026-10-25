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

    # The length of the Z axis is not defined
    with pytest.raises(NotImplementedError):
        Percent(Z_AXIS)

    # Degrees should convert from degrees to radians
    np.testing.assert_allclose(180 * Degrees, PI)


@pytest.mark.parametrize("axis", [X_AXIS, RIGHT, LEFT, 2 * LEFT, [-3, 0, 0]])
def test_percent_horizontal_directions_use_frame_width(config, axis):
    np.testing.assert_allclose(50 * Percent(axis), config.frame_width / 2)


@pytest.mark.parametrize("axis", [Y_AXIS, UP, DOWN, 0.5 * DOWN])
def test_percent_vertical_directions_use_frame_height(config, axis):
    np.testing.assert_allclose(50 * Percent(axis), config.frame_height / 2)


@pytest.mark.parametrize("axis", [UR, UL, [3, 1, 0], [1, 3, 0]])
def test_percent_diagonal_directions_use_extent_of_frame(config, axis):
    dx, dy = np.abs(axis[:2]) / np.linalg.norm(axis[:2])
    expected = min(config.frame_width / dx, config.frame_height / dy)
    np.testing.assert_allclose(100 * Percent(axis), expected)


def test_percent_diagonal_direction_explicit_value(config):
    # Along UR, the line through the centre leaves the frame through the top
    # and bottom edges, so the length is frame_height * sqrt(2).
    np.testing.assert_allclose(100 * Percent(UR), 8 * np.sqrt(2))


def test_percent_ignores_z_component(config):
    np.testing.assert_allclose(Percent(RIGHT + OUT) * 50, config.frame_width / 2)


@pytest.mark.parametrize("axis", [Z_AXIS, OUT, IN])
def test_percent_z_direction_is_not_implemented(axis):
    with pytest.raises(NotImplementedError):
        Percent(axis)


@pytest.mark.parametrize(
    "axis", [[0, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0], [1, np.nan, 0]]
)
def test_percent_rejects_invalid_vectors(axis):
    with pytest.raises(ValueError):
        Percent(axis)


def test_percent_follows_config_changes(config):
    percent = Percent(RIGHT)
    config.frame_width = 20
    np.testing.assert_allclose(50 * percent, 10)
