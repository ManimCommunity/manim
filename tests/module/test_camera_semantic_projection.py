"""Semantic camera queries and identity do not depend on drawing caches."""

import numpy as np

from manim import ThreeDCamera
from manim.utils.hashing import get_json


def test_projection_and_identity_do_not_depend_on_drawing():
    camera = ThreeDCamera()
    camera.set_theta(0)
    point = np.array([1.0, 0.0, 0.0])
    identity = get_json(camera)
    before = camera.project_point(point)
    np.testing.assert_allclose(before, [0, -1, 0], atol=1e-12)
    camera._prepare_for_render()
    np.testing.assert_array_equal(before, camera.project_point(point))
    assert get_json(camera) == identity
    camera.theta_tracker.set_value(0.4)
    assert get_json(camera) != identity
    assert not np.array_equal(before, camera.project_point(point))
