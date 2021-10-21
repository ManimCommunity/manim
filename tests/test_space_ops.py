import numpy as np

from manim import cartesian_to_spherical, spherical_to_cartesian


def test_polar_coords():
    a = np.array([1, 1, 0])
    b = (2, np.pi / 2, np.pi / 2)
    assert all(
        np.round(cartesian_to_spherical(a), 4)
        == np.round([2 ** 0.5, np.pi / 2, np.pi / 4], 4),
    )
    assert all(np.round(spherical_to_cartesian(b), 4) == np.array([0, 2, 0]))
