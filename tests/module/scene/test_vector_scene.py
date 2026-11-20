from __future__ import annotations

import numpy as np
import pytest

from manim import RED, Arrow, LinearTransformationScene, VectorScene


@pytest.mark.parametrize(
    ("plane_kwargs", "scale", "offset"),
    [
        pytest.param({}, 1, (0, 0, 0), id="default"),
        pytest.param({}, 1, (2, -1, 0), id="translated"),
        pytest.param({}, 0.5, (0, 0, 0), id="scaled"),
        pytest.param(
            {
                "x_range": (-4, 4, 2),
                "y_range": (-4, 4, 2),
                "x_length": 8,
                "y_length": 4,
            },
            0.8,
            (-1, 0.5, 0),
            id="transformed-non-unit-plane",
        ),
    ],
)
def test_get_vector_uses_added_plane(dry_run, plane_kwargs, scale, offset):
    scene = VectorScene()
    plane = scene.add_plane(**plane_kwargs)
    plane.scale(scale).shift(offset)
    original_mobjects = tuple(scene.mobjects)

    vector = scene.get_vector([1, 2, 0], color=RED)

    assert isinstance(vector, Arrow)
    assert vector.get_color() == RED
    np.testing.assert_allclose(vector.get_start(), plane.c2p(0, 0), atol=1e-7)
    np.testing.assert_allclose(vector.get_end(), plane.c2p(1, 2), atol=1e-7)
    assert scene.plane is plane
    assert tuple(scene.mobjects) == original_mobjects
    assert vector not in scene.mobjects


@pytest.mark.parametrize("transformed", [False, True])
def test_get_vector_preserves_linear_transformation_plane(dry_run, transformed):
    scene = LinearTransformationScene(show_basis_vectors=False)
    scene.setup()
    plane = scene.plane
    if transformed:
        plane.scale(0.5).shift((1, -1, 0))
    original_mobjects = tuple(scene.mobjects)
    original_transformable_mobjects = tuple(scene.transformable_mobjects)

    vector = scene.get_vector([1, 2, 0])

    np.testing.assert_allclose(vector.get_start(), plane.c2p(0, 0), atol=1e-7)
    np.testing.assert_allclose(vector.get_end(), plane.c2p(1, 2), atol=1e-7)
    assert scene.plane is plane
    assert tuple(scene.mobjects) == original_mobjects
    assert tuple(scene.transformable_mobjects) == original_transformable_mobjects
