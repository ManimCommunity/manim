from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "transform"


@frames_comparison(last_frame=False)
def test_Transform(scene):
    square = Square()
    circle = Circle()
    scene.play(Transform(square, circle))


@frames_comparison(last_frame=False)
def test_TransformFromCopy(scene):
    square = Square()
    circle = Circle()
    scene.play(TransformFromCopy(square, circle))


@frames_comparison(last_frame=False)
def test_FullRotation(scene):
    s = VGroup(*(Square() for _ in range(4))).arrange()
    scene.play(
        Rotate(s[0], -2 * TAU),
        Rotate(s[1], -1 * TAU),
        Rotate(s[2], 1 * TAU),
        Rotate(s[3], 2 * TAU),
    )


@frames_comparison(last_frame=False)
def test_ClockwiseTransform(scene):
    square = Square()
    circle = Circle()
    scene.play(ClockwiseTransform(square, circle))


@frames_comparison(last_frame=False)
def test_CounterclockwiseTransform(scene):
    square = Square()
    circle = Circle()
    scene.play(CounterclockwiseTransform(square, circle))


@frames_comparison(last_frame=False)
def test_MoveToTarget(scene):
    square = Square()
    square.generate_target()
    square.target.shift(3 * UP)
    scene.play(MoveToTarget(square))


@frames_comparison(last_frame=False)
def test_ApplyPointwiseFunction(scene):
    square = Square()

    def func(p):
        return np.array([1.0, 1.0, 0.0])

    scene.play(ApplyPointwiseFunction(func, square))


@frames_comparison(last_frame=False)
def test_FadeToColort(scene):
    square = Square()
    scene.play(FadeToColor(square, RED))


@frames_comparison(last_frame=False)
def test_ScaleInPlace(scene):
    square = Square()
    scene.play(ScaleInPlace(square, scale_factor=0.1))


@frames_comparison(last_frame=False)
def test_ShrinkToCenter(scene):
    square = Square()
    scene.play(ShrinkToCenter(square))


@frames_comparison(last_frame=False)
def test_Restore(scene):
    square = Square()
    circle = Circle()
    scene.play(Transform(square, circle))
    square.save_state()
    scene.play(square.animate.shift(UP))
    scene.play(Restore(square))


@frames_comparison
def test_ApplyFunction(scene):
    square = Square()
    scene.add(square)

    def apply_function(mob):
        mob.scale(2)
        mob.to_corner(UR)
        mob.rotate(PI / 4)
        mob.set_color(RED)
        return mob

    scene.play(ApplyFunction(apply_function, square))


@frames_comparison(last_frame=False)
def test_ApplyComplexFunction(scene):
    square = Square()
    scene.play(
        ApplyComplexFunction(
            lambda complex_num: complex_num + 2 * complex(0, 1),
            square,
        ),
    )


@frames_comparison(last_frame=False)
def test_ApplyMatrix(scene):
    square = Square()
    matrice = [[1.0, 0.5], [1.0, 0.0]]
    about_point = np.asarray((-10.0, 5.0, 0.0))
    scene.play(ApplyMatrix(matrice, square, about_point))


@frames_comparison(last_frame=False)
def test_CyclicReplace(scene):
    square = Square()
    circle = Circle()
    circle.shift(3 * UP)
    scene.play(CyclicReplace(square, circle))


@frames_comparison(last_frame=False)
def test_FadeInAndOut(scene):
    square = Square(color=BLUE).shift(2 * UP)
    annotation = Square(color=BLUE)
    scene.add(annotation)
    scene.play(FadeIn(square))

    annotation.become(Square(color=RED).rotate(PI / 4))
    scene.add(annotation)
    scene.play(FadeOut(square))


@frames_comparison
def test_MatchPointsScene(scene):
    circ = Circle(fill_color=RED, fill_opacity=0.8)
    square = Square(fill_color=BLUE, fill_opacity=0.2)
    scene.play(circ.animate.match_points(square))


@frames_comparison(last_frame=False)
def test_AnimationBuilder(scene):
    scene.play(Square().animate.shift(RIGHT).rotate(PI / 4))


@frames_comparison(last_frame=False)
def test_ReplacementTransform(scene):
    v1 = Vector()
    v2 = Vector()
    v3 = Line()
    scene.play(ReplacementTransform(VGroup(v1, v2), v3))


@frames_comparison(last_frame=False)
def test_TransformWithPathFunc(scene):
    dots_start = VGroup(*[Dot(LEFT, color=BLUE), Dot(3 * RIGHT, color=RED)])
    dots_end = VGroup(*[Dot(LEFT + 2 * DOWN, color=BLUE), Dot(2 * UP, color=RED)])
    scene.play(Transform(dots_start, dots_end, path_func=clockwise_path()))


@frames_comparison(last_frame=False)
def test_TransformWithPathArcCenters(scene):
    dots_start = VGroup(*[Dot(LEFT, color=BLUE), Dot(3 * RIGHT, color=RED)])
    dots_end = VGroup(*[Dot(LEFT + 2 * DOWN, color=BLUE), Dot(2 * UP, color=RED)])
    scene.play(
        Transform(
            dots_start,
            dots_end,
            path_arc=2 * PI,
            path_arc_centers=ORIGIN,
        )
    )


@frames_comparison(last_frame=False)
def test_TransformWithConflictingPaths(scene):
    dots_start = VGroup(*[Dot(LEFT, color=BLUE), Dot(3 * RIGHT, color=RED)])
    dots_end = VGroup(*[Dot(LEFT + 2 * DOWN, color=BLUE), Dot(2 * UP, color=RED)])
    scene.play(
        Transform(
            dots_start,
            dots_end,
            path_func=clockwise_path(),
            path_arc=2 * PI,
            path_arc_centers=ORIGIN,
        )
    )
