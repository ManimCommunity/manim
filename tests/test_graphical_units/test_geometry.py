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
        a = Arc(start_angle=PI)
        self.play(Animation(a))


class ArcBetweenPointsTest(Scene):
    def construct(self):
        a = ArcBetweenPoints(np.array([1, 1, 0]), np.array([2, 2, 0]))
        self.play(Animation(a))


class ArrowTest(Scene):
    def construct(self):
        a = Arrow()
        self.play(Animation(a))
        b = Arrow(color=RED, stroke_width=12, end=2*LEFT, start=2*RIGHT).scale(2).shift(UP)
        self.play(Animation(b))


class CurvedArrowTest(Scene):
    def construct(self):
        a = CurvedArrow(np.array([1, 1, 0]), np.array([2, 2, 0]))
        self.play(Animation(a))


class CustomDoubleArrowTest(Scene):
    def construct(self):
        from manim.mobject.geometry import ArrowCircleTip, ArrowSquareFilledTip

        a = DoubleArrow(
            start=np.array([-1, -1, 0]),
            end=np.array([1, 1, 0]),
            tip_shape_start=ArrowCircleTip,
            tip_shape_end=ArrowSquareFilledTip,
        )
        self.play(Animation(a))


class CircleTest(Scene):
    def construct(self):
        circle = Circle()
        self.play(Animation(circle))


class SmallDotTest(Scene):
    def construct(self):
        a = SmallDot()
        self.play(Animation(a))
        b = SmallDot(color=BLUE).shift(UP)
        self.play(Animation(b))


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


class DashedLineTest(Scene):
    def construct(self):
        a = DashedLine()
        self.play(Animation(a))


class CustomDashedLineTest(Scene):
    def construct(self):
        a = DashedLine(end=2*DOWN, start=2*UP, dash_length=0.5)
        self.play(Animation(a))


class TangentLineTest(Scene):
    def construct(self):
        circle = Circle(color=WHITE)
        self.add(circle)
        t1 = TangentLine(circle, 0.5, color=BLUE)
        self.play(Animation(t1))
        t2 = TangentLine(circle, 1.0, length=2, color=RED)
        self.play(Animation(t2))
        t3 = TangentLine(circle, 0.75, length=5, d_alpha=0.1, color=YELLOW)
        self.play(Animation(t3))
        t4 = TangentLine(circle, 0.321, length=1.5, d_alpha=-0.1, color=GREEN)
        self.play(Animation(t4))


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


MODULE_NAME = "geometry"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
