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


def _angles(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def test_vertices_at_one_point_of_a_smooth_surface_share_their_normal():
    from manim import Sphere, Torus
    from manim.renderer._mesh import vertex_normals

    torus = Torus(checkerboard_colors=False, resolution=(24, 24))
    nu, nv = torus.resolution
    normals = vertex_normals(torus.points, torus.triangles).reshape(nu + 1, nv + 1, 3)
    # Where the parameter grid closes, the first and last rows and columns meet.
    assert _angles(normals[0], normals[-1]).max() < 1e-3
    assert _angles(normals[:, 0], normals[:, -1]).max() < 1e-3
    # The exact normal of a torus at its outermost circle points outward.
    outer = torus.points[:, :2]
    distance = np.linalg.norm(outer, axis=1)
    rim = np.isclose(distance, distance.max())
    radial = np.zeros_like(torus.points)
    radial[:, :2] = outer / distance[:, None]
    assert _angles(normals.reshape(-1, 3)[rim], radial[rim]).max() < 1

    sphere = Sphere(checkerboard_colors=False)
    normals = vertex_normals(sphere.points, sphere.triangles)
    pole = np.isclose(sphere.points, [0, 0, 1]).all(axis=1)
    assert pole.sum() > 1
    np.testing.assert_allclose(normals[pole], [[0, 0, 1]] * pole.sum(), atol=1e-9)


def test_meshes_with_faces_of_their_own_keep_flat_normals():
    from manim import Cube, Torus
    from manim.renderer._mesh import vertex_normals

    # A checkerboard: every quad has its own corners.
    torus = Torus(resolution=(8, 8), checkerboard_colors=["#FF0000", "#0000FF"])
    normals = vertex_normals(torus.points, torus.triangles).reshape(-1, 4, 3)
    assert _angles(normals, normals[:, :1]).max() < 1e-3
    for face in Cube():
        normals = vertex_normals(face.points, face.triangles)
        assert _angles(normals, normals[:1]).max() < 1e-3


def test_accumulated_normals_are_those_computed_from_scratch():
    from manim import Cube, Sphere, Torus
    from manim.renderer._mesh import NormalAccumulator, vertex_normals

    meshes = [
        Torus(resolution=(12, 8)),
        Sphere(),
        Torus(resolution=(6, 6), checkerboard_colors=["#FF0000", "#0000FF"]),
        Cube()[0],
    ]
    for mesh in meshes:
        accumulator = NormalAccumulator(mesh.points, mesh.triangles)
        for angle in (0.0, 0.7):
            points = mesh.copy().rotate(angle, [1, 2, 3]).points
            np.testing.assert_allclose(
                accumulator.normals(points),
                vertex_normals(points, mesh.triangles),
                atol=1e-12,
            )


def test_an_accumulator_does_not_fit_once_welded_vertices_move_apart():
    from manim import UP, Sphere
    from manim.renderer._mesh import NormalAccumulator, vertex_normals

    sphere = Sphere(resolution=(8, 6))
    triangles = sphere.triangles
    accumulator = NormalAccumulator(sphere.points, triangles)
    moved = sphere.copy().rotate(0.4, UP).scale(3).shift(UP)
    assert accumulator.fits(moved.points, triangles)

    # The pole is one point of several vertices: pull one of them away.
    points = np.array(sphere.points)
    pole = np.flatnonzero(np.isclose(points, points[0]).all(axis=1))
    assert len(pole) > 1
    points[pole[-1]] += [0.3, 0.0, 0.0]

    assert not accumulator.fits(points, triangles)
    np.testing.assert_allclose(
        NormalAccumulator(points, triangles).normals(points),
        vertex_normals(points, triangles),
        atol=1e-12,
    )


def test_normals_of_a_deformed_mesh_are_not_those_of_its_old_welds():
    from manim import Sphere
    from manim.renderer import extract
    from manim.renderer._mesh import vertex_normals
    from manim.renderer.opengl import drawing

    cache = drawing._NormalCache()
    sphere = Sphere(resolution=(8, 6))
    node = extract.snapshot(sphere)
    cache.get(node, node.payload)
    cache.next_frame()
    points = np.array(sphere.points)
    pole = np.flatnonzero(np.isclose(points, points[0]).all(axis=1))
    points[pole[-1]] += [0.3, 0.0, 0.0]
    sphere.points = points

    node = extract.snapshot(sphere)

    np.testing.assert_allclose(
        cache.get(node, node.payload),
        vertex_normals(sphere.points, sphere.triangles),
        atol=1e-12,
    )


def test_normals_of_a_moving_mesh_reuse_its_accumulator(monkeypatch):
    from manim import UP, Torus
    from manim.renderer import _mesh, extract
    from manim.renderer.opengl import drawing

    made = []
    original = _mesh.NormalAccumulator
    monkeypatch.setattr(
        drawing,
        "NormalAccumulator",
        lambda *args: made.append(1) or original(*args),
    )
    cache = drawing._NormalCache()
    torus = Torus(resolution=(8, 8))
    for _ in range(3):
        node = extract.snapshot(torus)
        normals = cache.get(node, node.payload)
        cache.next_frame()
        torus.rotate(0.3, UP)
    assert len(made) == 1
    node = extract.snapshot(torus)
    np.testing.assert_allclose(
        cache.get(node, node.payload),
        _mesh.vertex_normals(node.payload.points, node.payload.triangles),
        atol=1e-12,
    )
    assert normals.shape == node.payload.points.shape
    # New triangles (as a partial surface shows): a new accumulator.
    torus.triangles = torus.triangles[: len(torus.triangles) // 2]
    node = extract.snapshot(torus)
    cache.get(node, node.payload)
    assert len(made) == 2
