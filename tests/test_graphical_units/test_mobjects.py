from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "mobjects"


@frames_comparison(base_scene=ThreeDScene)
def test_PointCloudDot(scene):
    p = PointCloudDot()
    scene.add(p)


@frames_comparison
def test_become(scene):
    s = Rectangle(width=2, height=1, color=RED).shift(UP)
    d1, d2, d3 = (Dot() for _ in range(3))

    s1 = s.copy().become(d1, match_width=True).set_opacity(0.25).set_color(BLUE)
    s2 = (
        s.copy()
        .become(d2, match_height=True, match_center=True)
        .set_opacity(0.25)
        .set_color(GREEN)
    )
    s3 = s.copy().become(d3, stretch=True).set_opacity(0.25).set_color(YELLOW)

    scene.add(s, d1, d2, d3, s1, s2, s3)
