import pytest
import numpy as np
import itertools
from manim import Sector, Vector, normalize


def test_get_arc_center():
    assert np.all(Sector(arc_center=[1, 2, 0]).get_arc_center() == [1, 2, 0])


def test_vector_tip_orientation():
    """Check if the orientation of Vector and its tip match"""
    # test vectors in all quadrants
    points = list(itertools.product([2.5, -3], repeat=3))
    vectors = list(map(Vector, points))
    for point, vector in zip(points, vectors):
        vector_direction = np.round(normalize(vector.get_vector()), 8)
        tip_direction = np.round(normalize(vector.tip.vector), 8)
        assert np.all(tip_direction == vector_direction)
        assert np.all(np.round(vector.tip.tip_point, 8) == np.round(point, 8))
