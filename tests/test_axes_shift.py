import numpy as np
import pytest

from manim.mobject.coordinate_systems import Axes, ThreeDAxes


def test_axes_origin_shift():
    ax = Axes(x_range=(5, 10, 1), y_range=(40, 45, 0.5))
    assert np.allclose(ax.coords_to_point(5, 40), ax.x_axis.number_to_point(5))
    assert np.allclose(ax.coords_to_point(5, 40), ax.y_axis.number_to_point(40))


def test_3daxes_origin_shift():
    ax = ThreeDAxes(x_range=(3, 9, 1), y_range=(6, 12, 1), z_range=(-1, 1, 0.5))
    assert np.allclose(ax.coords_to_point(3, 6, 0), ax.x_axis.number_to_point(3))
    assert np.allclose(ax.coords_to_point(3, 6, 0), ax.y_axis.number_to_point(6))
    assert np.allclose(ax.coords_to_point(3, 6, 0), ax.z_axis.number_to_point(0))
