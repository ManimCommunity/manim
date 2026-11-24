import math

import numpy as np
from scipy.spatial import cKDTree

from manim import Ellipse
from manim.renderer.opengl._drawing import cubics_to_quadratics


def _samples(controls: np.ndarray, count: int) -> np.ndarray:
    t = np.linspace(0, 1, count)[:, None]
    degree = len(controls) - 1
    weights = [
        math.comb(degree, i) * (1 - t) ** (degree - i) * t**i for i in range(degree + 1)
    ]
    return sum(w * c for w, c in zip(weights, controls, strict=True))


def test_quadratic_pieces_stay_within_tolerance():
    cubics = Ellipse(6, 1).points
    for tolerance in (0.01, 0.001):
        quads = cubics_to_quadratics(cubics, tolerance)
        curve = np.concatenate([_samples(c, 400) for c in cubics.reshape(-1, 4, 3)])
        pieces = np.concatenate([_samples(q, 4000) for q in quads])
        distance, _ = cKDTree(pieces).query(curve)
        assert distance.max() <= tolerance
        np.testing.assert_allclose(quads[0, 0], cubics[0])
        np.testing.assert_allclose(quads[-1, 2], cubics[-1])
    assert len(cubics_to_quadratics(cubics, 0.001)) > len(
        cubics_to_quadratics(cubics, 0.01)
    )


def test_cached_geometry_follows_the_point_tolerance():
    from manim import VMobject
    from manim.renderer.extract import snapshot
    from manim.renderer.opengl._drawing import _GeometryCache

    # Two curves whose ends are 1e-3 apart: one subpath or two, by the tolerance.
    path = VMobject()
    path.points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        + [[0, 1.001, 0], [-1, 1, 0], [-1, 0, 0], [0, 0, 0]],
        dtype=float,
    )

    def subpath_starts(cache):
        cache.next_frame()
        return sorted(set(cache.get([snapshot(path)], 0.01)[0].subpath_start.tolist()))

    cache = _GeometryCache()
    assert len(subpath_starts(cache)) == 2

    path.tolerance_for_point_equality = 0.01
    assert subpath_starts(cache) == subpath_starts(_GeometryCache()) == [0]
