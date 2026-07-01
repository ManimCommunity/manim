from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "polyhedra"


@frames_comparison
def test_Tetrahedron(scene):
    scene.add(Tetrahedron())


@frames_comparison
def test_Octahedron(scene):
    scene.add(Octahedron())


@frames_comparison
def test_Icosahedron(scene):
    scene.add(Icosahedron())


@frames_comparison
def test_Dodecahedron(scene):
    scene.add(Dodecahedron())


@frames_comparison
def test_ConvexHull3D(scene):
    a = ConvexHull3D(
        *[
            [-2.7, -0.6, 3.5],
            [0.2, -1.7, -2.8],
            [1.9, 1.2, 0.7],
            [-2.7, 0.9, 1.9],
            [1.6, 2.2, -4.2],
        ]
    )
    scene.add(a)
