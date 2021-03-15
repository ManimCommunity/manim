import pytest

from manim import *
from ..utils.testing_utils import get_scenes_to_test
from ..utils.GraphicalUnitTester import GraphicalUnitTester


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


class CustomDoubleArrowTest(Scene):
    def construct(self):
        from manim.mobject.geometry import ArrowCircleTip, ArrowSquareFilledTip

        a = DoubleArrow(
            np.array([-1, -1, 0]),
            np.array([1, 1, 0]),
            tip_shape_start=ArrowCircleTip,
            tip_shape_end=ArrowSquareFilledTip,
        )
        self.play(Animation(a))


class CircleTest(Scene):
    def construct(self):
        circle = Circle()
        self.play(Animation(circle))


class DotTest(Scene):
    def construct(self):
        dot = Dot()
        self.play(Animation(dot))


class AnnotationDotTest(Scene):
    def construct(self):
        adot = AnnotationDot()
        self.play(Animation(adot))


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


class ElbowTest(Scene):
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
        a = Polygon(*[np.array([1, 1, 0]), np.array([2, 2, 0]), np.array([2, 3, 0])])
        self.play(Animation(a))


class RectangleTest(Scene):
    def construct(self):
        a = Rectangle()
        self.play(Animation(a))


class RoundedRectangleTest(Scene):
    def construct(self):
        a = RoundedRectangle()
        self.play(Animation(a))


class ArrangeTest(Scene):
    def construct(self):
        s1 = Square()
        s2 = Square()
        x = VGroup(s1, s2).set_x(0).arrange(buff=1.4)
        self.add(x)
        self.wait()


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
        self.wait(1)


class AngleTest(Scene):
    def construct(self):
        l1 = Line(ORIGIN, RIGHT)
        l2 = Line(ORIGIN, UP)
        a = Angle(l1, l2)
        self.play(Animation(a))


class RightAngleTest(Scene):
    def construct(self):
        l1 = Line(ORIGIN, RIGHT)
        l2 = Line(ORIGIN, UP)
        a = RightAngle(l1, l2)
        self.play(Animation(a))


MODULE_NAME = "geometry"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
