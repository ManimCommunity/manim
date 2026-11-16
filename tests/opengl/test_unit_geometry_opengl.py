from __future__ import annotations

import numpy as np

from manim import Sector


def test_get_arc_center(using_opengl_renderer):
    np.testing.assert_array_equal(
        Sector(arc_center=[1, 2, 0]).get_arc_center(), [1, 2, 0]
    )
