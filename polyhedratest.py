from itertools import permutations

from manim import *


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


class IcosahedronTest(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        a = Icosahedron(side_length=3)
        self.add(a)
        self.play(a.graph[0].animate.move_to(ORIGIN))
        self.wait()


class DodecahedronTest(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        a = Dodecahedron(side_length=3)
        self.add(a)


class UpdatersTest(ThreeDScene):
    def construct(self):
        # self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        vertices = [[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]]
        faces = [[0, 1, 2, 3]]
        a = Polyhedron(vertices, faces)
        self.add(a)
        self.remove(a.faces)
        self.play(a.graph[0].animate.move_to(ORIGIN))
        self.wait()


class MovingVertices(ThreeDScene):
    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        vertices = [1, 2, 3, 4]
        edges = [(1, 2), (2, 3), (3, 4), (1, 3), (1, 4)]
        g = Graph(vertices, edges)
        self.play(Create(g))
        self.wait()
        self.play(
            g[1].animate.move_to([1, 1, 1]),
            g[2].animate.move_to([-1, 1, 2]),
            g[3].animate.move_to([1, -1, -1]),
            g[4].animate.move_to([-1, -1, 0]),
        )
        self.wait()


class MovingVertices2(Scene):
    def construct(self):
        vertices = [1, 2, 3, 4]
        edges = [(1, 2), (2, 3), (3, 4), (1, 3), (1, 4)]
        g = Graph(vertices, edges)
        self.play(Create(g))
        self.wait()
        self.play(
            g[1].animate.move_to([1, 1, 0]),
            g[2].animate.move_to([-1, 1, 0]),
            g[3].animate.move_to([1, -1, 0]),
            g[4].animate.move_to([-1, -1, 0]),
        )
        self.wait()
        for vertex in g:
            print(vertex.get_center())
        print("g.vertices: ", g._layout)
        self.wait()
