from __future__ import annotations

import numpy as np
import pytest

from manim import (
    DOWN,
    IN,
    LEFT,
    OUT,
    PI,
    RIGHT,
    UL,
    UP,
    X_AXIS,
    Y_AXIS,
    Z_AXIS,
)
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


@pytest.mark.parametrize("axis", [X_AXIS, RIGHT, LEFT])
def test_percent_horizontal_directions_use_frame_width(config, axis):
    np.testing.assert_allclose(50 * Percent(axis), config.frame_width / 2)
    np.testing.assert_allclose(Percent(axis) * 25, config.frame_width / 4)


@pytest.mark.parametrize("axis", [Y_AXIS, UP, DOWN])
def test_percent_vertical_directions_use_frame_height(config, axis):
    np.testing.assert_allclose(50 * Percent(axis), config.frame_height / 2)
    np.testing.assert_allclose(Percent(axis) * 25, config.frame_height / 4)


@pytest.mark.parametrize("axis", [Z_AXIS, OUT, IN])
def test_percent_z_directions_are_not_implemented(axis):
    with pytest.raises(NotImplementedError):
        Percent(axis)


def test_percent_rejects_non_axis_vectors():
    with pytest.raises(ValueError, match="X, Y or Z axis"):
        Percent(UL)
