from __future__ import annotations

import numpy as np
import pytest

from manim import PI, X_AXIS, Y_AXIS, Z_AXIS, config
from manim.utils.unit import Degrees, Munits, Percent, Pixels


def test_units():
    # make sure we are using the right frame geometry
    assert config.pixel_width == 1920

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
