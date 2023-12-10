from __future__ import annotations

import numpy as np
import numpy.testing as nt

from manim.utils.color import BLACK, WHITE, ManimColor, ManimColorDType


def test_init_with_int() -> None:
    color = ManimColor(0x123456, 0.5)
    nt.assert_array_equal(
        color._internal_value,
        np.array([0x12, 0x34, 0x56, 0.5 * 255], dtype=ManimColorDType) / 255,
    )
    color = BLACK
    nt.assert_array_equal(
        color._internal_value, np.array([0, 0, 0, 1.0], dtype=ManimColorDType)
    )
    color = WHITE
    nt.assert_array_equal(
        color._internal_value, np.array([1.0, 1.0, 1.0, 1.0], dtype=ManimColorDType)
    )
