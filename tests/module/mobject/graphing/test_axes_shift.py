from __future__ import annotations

import numpy as np

from manim.mobject.graphing.coordinate_systems import Axes, ThreeDAxes
from manim.mobject.graphing.scale import LogBase


def test_axes_origin_shift():
    ax = Axes(x_range=(5, 10, 1), y_range=(40, 45, 0.5))
    np.testing.assert_allclose(ax.coords_to_point(5, 40), ax.x_axis.number_to_point(5))
    np.testing.assert_allclose(ax.coords_to_point(5, 40), ax.y_axis.number_to_point(40))


def test_axes_origin_shift_logbase():
    ax = Axes(
        x_range=(5, 10, 1),
        y_range=(3, 8, 1),
        x_axis_config={"scaling": LogBase()},
        y_axis_config={"scaling": LogBase()},
    )
    np.testing.assert_allclose(
        ax.coords_to_point(10**5, 10**3), ax.x_axis.number_to_point(10**5)
    )
    np.testing.assert_allclose(
        ax.coords_to_point(10**5, 10**3), ax.y_axis.number_to_point(10**3)
    )


def test_3daxes_origin_shift():
    ax = ThreeDAxes(x_range=(3, 9, 1), y_range=(6, 12, 1), z_range=(-1, 1, 0.5))
    np.testing.assert_allclose(
        ax.coords_to_point(3, 6, 0), ax.x_axis.number_to_point(3)
    )
    np.testing.assert_allclose(
        ax.coords_to_point(3, 6, 0), ax.y_axis.number_to_point(6)
    )
    np.testing.assert_allclose(
        ax.coords_to_point(3, 6, 0), ax.z_axis.number_to_point(0)
    )


def test_3daxes_origin_shift_logbase():
    ax = ThreeDAxes(
        x_range=(3, 9, 1),
        y_range=(6, 12, 1),
        z_range=(2, 5, 1),
        x_axis_config={"scaling": LogBase()},
        y_axis_config={"scaling": LogBase()},
        z_axis_config={"scaling": LogBase()},
    )
    np.testing.assert_allclose(
        ax.coords_to_point(10**3, 10**6, 10**2),
        ax.x_axis.number_to_point(10**3),
    )
    np.testing.assert_allclose(
        ax.coords_to_point(10**3, 10**6, 10**2),
        ax.y_axis.number_to_point(10**6),
    )
    np.testing.assert_allclose(
        ax.coords_to_point(10**3, 10**6, 10**2),
        ax.z_axis.number_to_point(10**2),
    )
