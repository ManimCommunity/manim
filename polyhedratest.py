from manim import *
from itertools import permutations

class Test(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        vertices = [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5],
        ]
        faces = [
            [4, 0, 2, 6],
            [0, 1, 3, 2],
            [6, 7, 3, 2],
            [5, 7, 6, 4],
            [4, 0, 1, 5],
            [7, 5, 1, 3],
        ]
        a = Polyhedra(vertices, faces)
        self.add(a)
        self.remove(a.faces)

class TetrahedronTest(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        a = Tetrahedron()
        self.add(a)

class OctahedronTest(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        a = Octahedron(side_length=3)
        self.add(a)
