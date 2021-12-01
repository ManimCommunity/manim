from manim import (
    BLUE,
    Circle,
    Difference,
    Exclusion,
    Intersection,
    Rectangle,
    Square,
    Triangle,
    Union,
)

# not exported by default, so directly import
from manim.mobject.boolean_ops import _BooleanOps
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "boolean_ops"


@frames_comparison()
def test_union(scene):
    a = Square()
    b = Circle().move_to([0.2, 0.2, 0.0])
    c = Rectangle()
    un = Union(a, b, c).next_to(b)
    scene.add(a, b, c, un)


@frames_comparison()
def test_intersection(scene):
    a = Square()
    b = Circle().move_to([0.3, 0.3, 0.0])
    i = Intersection(a, b).next_to(b)
    scene.add(a, b, i)


@frames_comparison()
def test_difference(scene):
    a = Square()
    b = Circle().move_to([0.2, 0.3, 0.0])
    di = Difference(a, b).next_to(b)
    scene.add(a, b, di)


@frames_comparison()
def test_exclusion(scene):
    a = Square()
    b = Circle().move_to([0.3, 0.2, 0.0])
    ex = Exclusion(a, b).next_to(a)
    scene.add(a, b, ex)


@frames_comparison()
def test_intersection_3_mobjects(scene):
    a = Square()
    b = Circle().move_to([0.2, 0.2, 0])
    c = Triangle()
    i = Intersection(a, b, c, fill_opacity=0.5, color=BLUE)
    scene.add(a, b, c, i)
