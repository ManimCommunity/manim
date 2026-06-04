from __future__ import annotations

import numpy as np

from manim.mobject.graphing.coordinate_systems import Axes


def test_axes_origin_shift(using_opengl_renderer):
    ax = Axes(x_range=(5, 10, 1), y_range=(40, 45, 0.5))
    np.testing.assert_allclose(
        ax.coords_to_point(5.0, 40.0), ax.x_axis.number_to_point(5)
    )
    np.testing.assert_allclose(
        ax.coords_to_point(5.0, 40.0), ax.y_axis.number_to_point(40)
    )
