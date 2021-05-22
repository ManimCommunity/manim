import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class CoordinatesTest(Scene):
    def construct(self):
        dots = [Dot(np.array([x, y, 0])) for x in range(-7, 8) for y in range(-4, 5)]
        self.add(VGroup(*dots))


class ArcTest(Scene):
    def construct(self):
        a = Arc(radius=1, start_angle=PI)
        self.add(a)


class ArcBetweenPointsTest(Scene):
    def construct(self):
        a = ArcBetweenPoints(np.array([1, 1, 0]), np.array([2, 2, 0]))
        self.add(a)


class CurvedArrowTest(Scene):
    def construct(self):
        a = CurvedArrow(np.array([1, 1, 0]), np.array([2, 2, 0]))
        self.add(a)


class CustomDoubleArrowTest(Scene):
    def construct(self):
        from manim.mobject.geometry import ArrowCircleTip, ArrowSquareFilledTip

        a = DoubleArrow(
            np.array([-1, -1, 0]),
            np.array([1, 1, 0]),
            tip_shape_start=ArrowCircleTip,
            tip_shape_end=ArrowSquareFilledTip,
        )
        self.add(a)


class CircleTest(Scene):
    def construct(self):
        circle = Circle()
        self.add(circle)


class DotTest(Scene):
    def construct(self):
        dot = Dot()
        self.add(dot)


class AnnotationDotTest(Scene):
    def construct(self):
        adot = AnnotationDot()
        self.add(adot)


class EllipseTest(Scene):
    def construct(self):
        e = Ellipse()
        self.add(e)


class SectorTest(Scene):
    def construct(self):
        e = Sector()
        self.add(e)


class AnnulusTest(Scene):
    def construct(self):
        a = Annulus()
        self.add(a)


class AnnularSectorTest(Scene):
    def construct(self):
        a = AnnularSector()
        self.add(a)


class LineTest(Scene):
    def construct(self):
        a = Line(np.array([1, 1, 0]), np.array([2, 2, 0]))
        self.add(a)


class ElbowTest(Scene):
    def construct(self):
        a = Elbow()
        self.add(a)


class DoubleArrowTest(Scene):
    def construct(self):
        a = DoubleArrow()
        self.add(a)


class VectorTest(Scene):
    def construct(self):
        a = Vector(UP)
        self.add(a)


class PolygonTest(Scene):
    def construct(self):
        a = Polygon(*[np.array([1, 1, 0]), np.array([2, 2, 0]), np.array([2, 3, 0])])
        self.add(a)


class RectangleTest(Scene):
    def construct(self):
        a = Rectangle()
        self.add(a)


class RoundedRectangleTest(Scene):
    def construct(self):
        a = RoundedRectangle()
        self.add(a)


class ArrangeTest(Scene):
    def construct(self):
        s1 = Square()
        s2 = Square()
        x = VGroup(s1, s2).set_x(0).arrange(buff=1.4)
        self.add(x)


class ZIndexTest(Scene):
    def construct(self):
        circle = Circle().set_fill(RED, opacity=1)
        square = Square(side_length=1.7).set_fill(BLUE, opacity=1)
        triangle = Triangle().set_fill(GREEN, opacity=1)
        square.z_index = 0
        triangle.z_index = 1
        circle.z_index = 2

        self.play(FadeIn(VGroup(circle, square, triangle)))
        self.play(ApplyMethod(circle.shift, UP))
        self.play(ApplyMethod(triangle.shift, 2 * UP))


class AngleTest(Scene):
    def construct(self):
        l1 = Line(ORIGIN, RIGHT)
        l2 = Line(ORIGIN, UP)
        a = Angle(l1, l2)
        self.add(a)


class RightAngleTest(Scene):
    def construct(self):
        l1 = Line(ORIGIN, RIGHT)
        l2 = Line(ORIGIN, UP)
        a = RightAngle(l1, l2)
        self.add(a)


class PolygramTest(Scene):
    def construct(self):
        hexagram = Polygram(
            [[0, 2, 0], [-np.sqrt(3), -1, 0], [np.sqrt(3), -1, 0]],
            [[-np.sqrt(3), 1, 0], [0, -2, 0], [np.sqrt(3), 1, 0]],
        )
        self.add(hexagram)


class RegularPolygramTest(Scene):
    def construct(self):
        pentagram = RegularPolygram(5, radius=2)
        octagram = RegularPolygram(8, radius=2)
        self.add(VGroup(pentagram, octagram).arrange(RIGHT))


class StarTest(Scene):
    def construct(self):
        star = Star(outer_radius=2)
        self.add(star)


MODULE_NAME = "geometry"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
