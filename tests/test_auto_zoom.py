from manim import *

from .simple_zoom_scene import test_zoom_pan_to_center


def test_zoom():

    s1 = Square()
    s1.set_x(-10)
    s2 = Square()
    s2.set_x(10)

    scene = test_zoom_pan_to_center()

    scene.construct()

    assert (
        scene.camera.frame_width
        == abs(
            s1.get_left()[0] - s2.get_right()[0],
        )
        and scene.camera.frame.get_center()[0]
        == (abs(s1.get_center()[0] + s2.get_center()[0]) / 2)
    )
