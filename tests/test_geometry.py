from manim import *
import numpy as np
from testing_utils import utils_test_scenes, get_scenes_to_test


class CoordinatesTest(Scene):
    def construct(self):
        dots = [Dot(np.array([x, y, 0])) for x in range(-7, 8) for y in range(-4, 5)]
        self.play(Animation(VGroup(*dots)))


class ArcTest(Scene):
    def construct(self):
        a = Arc(PI)
        self.play(Animation(a))


class ArcBetweenPointsTest(Scene):
    def construct(self):
        a = ArcBetweenPoints(np.array([1, 1, 0]), np.array([2, 2, 0]))
        self.play(Animation(a))


class CurvedArrowTest(Scene):
    def construct(self):
        a = CurvedArrow(np.array([1, 1, 0]), np.array([2, 2, 0]))
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
        a = Line(np.array([1, 1, 0]), np.array([2, 2, 0]))
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
        a = Vector(UP)
        self.play(Animation(a))


class PolygonTest(Scene):
    def construct(self):
        a = Polygon(
            *[np.array([1, 1, 0]), np.array([2, 2, 0]), np.array([2, 3, 0])])
        self.play(Animation(a))


class RectangleTest(Scene):
    def construct(self):
        a = Rectangle()
        self.play(Animation(a))


class RoundedRectangleTest(Scene):
    def construct(self):
        a = RoundedRectangle()
        self.play(Animation(a))


class ArcPolygonTest(Scene):
    def construct(self):
        r3 = np.sqrt(3)
        arc_config = {"stroke_width":3,"stroke_color":RED,
                      "fill_opacity":0.5,"color": GREEN}
        pol_config = {"stroke_width":10,"stroke_color":BLUE,
                      "fill_opacity":1,"color": PURPLE}
        arc0=ArcBetweenPoints(np.array([-1,0,0]),
                              np.array([1,0,0]),radius=2,**arc_config)
        arc1=ArcBetweenPoints(np.array([1,0,0]),
                              np.array([0,r3,0]),radius=2,**arc_config)
        arc2=ArcBetweenPoints(np.array([0,r3,0]),
                              np.array([-1,0,0]),radius=2,**arc_config)
        a=ArcPolygon(arc0,arc1,arc2,**pol_config)
        self.play(Animation(a))


class TilingTest(Scene):
    def construct(self):
        a=Tiling(Square(),
                    [[Mobject.shift,[2.1,0,0]]],
                    [[Mobject.shift,[0,2.1,0]]],
                    range(-1,1),
                    range(-1,1))
        self.play(Animation(a))

        
class HoneycombTest(ThreeDScene):
    def construct(self):
        a=Honeycomb(Cube(),
                    [[Mobject.shift,[2.1,0,0]]],
                    [[Mobject.shift,[0,2.1,0]]],
                    [[Mobject.shift,[0,0,2.1]]],
                    range(-1,1),
                    range(-1,1),
                    range(-1,1))
        self.play(Animation(a))


class GraphTest(Scene):
    def construct(self):
        g = {0: [[0,0,0], [[1,{"angle": 2}], [2,{"color": WHITE}]], {"color": BLUE}],
             1: [[1,0,0], [0, 2], {"color": GRAY}],
             2: [[0,1,0], [0, 1], {"color": PINK}]}
        K3 = Graph(g,vertex_config={"radius": 0.2,"fill_opacity": 1},
                   edge_config={"stroke_width": 5,"color": RED})
        K3v = K3.vertices
        K3e = K3.edges
        self.play(Animation(K3e),Animation(K3v))


def test_scenes():
    utils_test_scenes(get_scenes_to_test(__name__), "geometry")
