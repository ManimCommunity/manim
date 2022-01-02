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
