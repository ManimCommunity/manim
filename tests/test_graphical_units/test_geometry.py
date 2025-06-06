from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

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
    circle = DashedVMobject(Circle(), 12, 0.9, dash_offset=0.1)
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
def test_ConvexHull(scene):
    a = ConvexHull(
        *[
            [-2.7, -0.6, 0],
            [0.2, -1.7, 0],
            [1.9, 1.2, 0],
            [-2.7, 0.9, 0],
            [1.6, 2.2, 0],
        ]
    )
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
def test_three_points_Angle(scene):
    # acute angle
    acute = Angle.from_three_points(
        np.array([10, 0, 0]), np.array([0, 0, 0]), np.array([10, 10, 0])
    )
    # obtuse angle
    obtuse = Angle.from_three_points(
        np.array([-10, 1, 0]), np.array([0, 0, 0]), np.array([10, 1, 0])
    )
    # quadrant 1 angle
    q1 = Angle.from_three_points(
        np.array([10, 10, 0]), np.array([0, 0, 0]), np.array([10, 1, 0])
    )
    # quadrant 2 angle
    q2 = Angle.from_three_points(
        np.array([-10, 1, 0]), np.array([0, 0, 0]), np.array([-1, 10, 0])
    )
    # quadrant 3 angle
    q3 = Angle.from_three_points(
        np.array([-10, -1, 0]), np.array([0, 0, 0]), np.array([-1, -10, 0])
    )
    # quadrant 4 angle
    q4 = Angle.from_three_points(
        np.array([10, -1, 0]), np.array([0, 0, 0]), np.array([1, -10, 0])
    )
    scene.add(VGroup(acute, obtuse, q1, q2, q3, q4).arrange(RIGHT))


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


@frames_comparison
def test_LabeledLine(scene):
    line = LabeledLine(
        label="0.5",
        label_position=0.8,
        label_config={"font_size": 20},
        start=LEFT + DOWN,
        end=RIGHT + UP,
    )
    scene.add(line)


@frames_comparison
def test_LabeledArrow(scene):
    l_arrow = LabeledArrow(
        label="0.5",
        label_position=0.5,
        label_config={"font_size": 15},
        start=LEFT * 3,
        end=RIGHT * 3 + UP * 2,
    )
    scene.add(l_arrow)


@frames_comparison
def test_LabeledPolygram(scene):
    polygram = LabeledPolygram(
        [
            [-2.5, -2.5, 0],
            [2.5, -2.5, 0],
            [2.5, 2.5, 0],
            [-2.5, 2.5, 0],
            [-2.5, -2.5, 0],
        ],
        [[-1, -1, 0], [0.5, -1, 0], [0.5, 0.5, 0], [-1, 0.5, 0], [-1, -1, 0]],
        [[1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0], [1, 1, 0]],
        label="C",
    )
    scene.add(polygram)
