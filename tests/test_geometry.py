from manim import *
import numpy as np
from testing_utils import utils_test_scenes, get_scenes_to_test


class CoordinatesTest(Scene):
    def construct(self):
        dots = [Dot(point=np.array([x, y, 0])) for x in range(-7, 8) for y in range(-4, 5)]
        self.play(Animation(VGroup.from_vmobjects(*dots)))


class ArcTest(Scene):
    def construct(self):
        a = Arc(start_angle=PI)
        self.play(Animation(a))


class ArcBetweenPointsTest(Scene):
    def construct(self):
        a = ArcBetweenPoints(start=np.array([1, 1, 0]), end=np.array([2, 2, 0]))
        self.play(Animation(a))


class CurvedArrowTest(Scene):
    def construct(self):
        a = CurvedArrow(start_point=np.array([1, 1, 0]), end_point=np.array([2, 2, 0]))
        self.play(Animation(a))


class CircleTest(Scene):
    def construct(self):
        circle = Circle()
        self.play(Animation(circle))


class DotTest(Scene):
    def construct(self):
        dot = Dot()
        self.play(Animation(dot))


class EllipseTest(Scene):
    def construct(self):
        e = Ellipse()
        self.play(Animation(e))


class SectorTest(Scene):
    def construct(self):
        e = Sector()
        self.play(Animation(e))


class AnnulusTest(Scene):
    def construct(self):
        a = Annulus()
        self.play(Animation(a))


class AnnularSectorTest(Scene):
    def construct(self):
        a = AnnularSector()
        self.play(Animation(a))


class LineTest(Scene):
    def construct(self):
        a = Line(start=np.array([1, 1, 0]), end=np.array([2, 2, 0]))
        self.play(Animation(a))


class Elbowtest(Scene):
    def construct(self):
        a = Elbow()
        self.play(Animation(a))


class DoubleArrowTest(Scene):
    def construct(self):
        a = DoubleArrow()
        self.play(Animation(a))


class VectorTest(Scene):
    def construct(self):
        a = Vector(direction=UP)
        self.play(Animation(a))


class PolygonTest(Scene):
    def construct(self):
        a = Polygon(vertices=[np.array([1, 1, 0]), np.array([2, 2, 0]), np.array([2, 3, 0])])
        self.play(Animation(a))


class RectangleTest(Scene):
    def construct(self):
        a = Rectangle()
        self.play(Animation(a))


class RoundedRectangleTest(Scene):
    def construct(self):
        a = RoundedRectangle()
        self.play(Animation(a))


def test_scenes():
    utils_test_scenes(get_scenes_to_test(__name__), "geometry")
