from manim import *
from tests.test_graphical_units.testing.frames_comparison import frames_comparison

__module_test__ = "geometry"


@frames_comparison(last_frame=True)
def test_Coordinates(scene):
    dots = [Dot(np.array([x, y, 0])) for x in range(-7, 8) for y in range(-4, 5)]
    scene.add(VGroup(*dots))


@frames_comparison
def test_Arc(scene):
    a = Arc(radius=1, start_angle=PI)
    scene.add(a)


@frames_comparison
def test_ArcBetweenPoints(scene):
    a = ArcBetweenPoints(np.array([1, 1, 0]), np.array([2, 2, 0]))
    scene.add(a)


@frames_comparison
def test_CurvedArrow(scene):
    a = CurvedArrow(np.array([1, 1, 0]), np.array([2, 2, 0]))
    scene.add(a)


@frames_comparison
def test_CustomDoubleArrow(scene):
    a = DoubleArrow(
        np.array([-1, -1, 0]),
        np.array([1, 1, 0]),
        tip_shape_start=ArrowCircleTip,
        tip_shape_end=ArrowSquareFilledTip,
    )
    scene.add(a)


@frames_comparison
def test_Circle(scene):
    circle = Circle()
    scene.add(circle)


@frames_comparison
def test_CirclePoints(scene):
    circle = Circle.from_three_points(LEFT, LEFT + UP, UP * 2)
    scene.add(circle)


@frames_comparison
def test_Dot(scene):
    dot = Dot()
    scene.add(dot)


@frames_comparison
def test_DashedVMobject(scene):
    circle = DashedVMobject(Circle(), 12, 0.9)
    line = DashedLine(dash_length=0.5)
    scene.add(circle, line)


@frames_comparison
def test_AnnotationDot(scene):
    adot = AnnotationDot()
    scene.add(adot)


@frames_comparison
def test_Ellipse(scene):
    e = Ellipse()
    scene.add(e)


@frames_comparison
def test_Sector(scene):
    e = Sector()
    scene.add(e)


@frames_comparison
def test_Annulus(scene):
    a = Annulus()
    scene.add(a)


@frames_comparison
def test_AnnularSector(scene):
    a = AnnularSector()
    scene.add(a)


@frames_comparison
def test_Line(scene):
    a = Line(np.array([1, 1, 0]), np.array([2, 2, 0]))
    scene.add(a)


@frames_comparison
def test_Elbow(scene):
    a = Elbow()
    scene.add(a)


@frames_comparison
def test_DoubleArrow(scene):
    a = DoubleArrow()
    scene.add(a)


@frames_comparison
def test_Vector(scene):
    a = Vector(UP)
    scene.add(a)


@frames_comparison
def test_Polygon(scene):
    a = Polygon(*[np.array([1, 1, 0]), np.array([2, 2, 0]), np.array([2, 3, 0])])
    scene.add(a)


@frames_comparison
def test_Rectangle(scene):
    a = Rectangle()
    scene.add(a)


@frames_comparison
def test_RoundedRectangle(scene):
    a = RoundedRectangle()
    scene.add(a)


@frames_comparison
def test_Arrange(scene):
    s1 = Square()
    s2 = Square()
    x = VGroup(s1, s2).set_x(0).arrange(buff=1.4)
    scene.add(x)


@frames_comparison(last_frame=False)
def test_ZIndex(scene):
    circle = Circle().set_fill(RED, opacity=1)
    square = Square(side_length=1.7).set_fill(BLUE, opacity=1)
    triangle = Triangle().set_fill(GREEN, opacity=1)
    square.z_index = 0
    triangle.z_index = 1
    circle.z_index = 2

    scene.play(FadeIn(VGroup(circle, square, triangle)))
    scene.play(ApplyMethod(circle.shift, UP))
    scene.play(ApplyMethod(triangle.shift, 2 * UP))


@frames_comparison
def test_Angle(scene):
    l1 = Line(ORIGIN, RIGHT)
    l2 = Line(ORIGIN, UP)
    a = Angle(l1, l2)
    scene.add(a)


@frames_comparison
def test_RightAngle(scene):
    l1 = Line(ORIGIN, RIGHT)
    l2 = Line(ORIGIN, UP)
    a = RightAngle(l1, l2)
    scene.add(a)


@frames_comparison
def test_Polygram(scene):
    hexagram = Polygram(
        [[0, 2, 0], [-np.sqrt(3), -1, 0], [np.sqrt(3), -1, 0]],
        [[-np.sqrt(3), 1, 0], [0, -2, 0], [np.sqrt(3), 1, 0]],
    )
    scene.add(hexagram)


@frames_comparison
def test_RegularPolygram(scene):
    pentagram = RegularPolygram(5, radius=2)
    octagram = RegularPolygram(8, radius=2)
    scene.add(VGroup(pentagram, octagram).arrange(RIGHT))


@frames_comparison
def test_Star(scene):
    star = Star(outer_radius=2)
    scene.add(star)


@frames_comparison
def test_AngledArrowTip(scene):
    arrow = Arrow(start=ORIGIN, end=UP + RIGHT + OUT)
    scene.add(arrow)


@frames_comparison
def test_CurvedArrowCustomTip(scene):
    arrow = CurvedArrow(
        LEFT,
        RIGHT,
        tip_shape=ArrowCircleTip,
    )
    double_arrow = CurvedDoubleArrow(
        LEFT,
        RIGHT,
        tip_shape_start=ArrowCircleTip,
        tip_shape_end=ArrowSquareFilledTip,
    )
    scene.add(arrow, double_arrow)
