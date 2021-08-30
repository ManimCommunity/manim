import numpy as np
import pytest

from manim.mobject.coordinate_systems import Axes


def test_axes_origin_shift():
    ax = Axes(x_range=(5, 10, 1), y_range=(40, 45, 0.5))
    assert np.allclose(ax.coords_to_point(5, 40), ax.x_axis.number_to_point(5))
    assert np.allclose(ax.coords_to_point(5, 40), ax.y_axis.number_to_point(40))
