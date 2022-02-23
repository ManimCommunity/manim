from __future__ import annotations

from manim import *


def test_zoom():

    s1 = Square()
    s1.set_x(-10)
    s2 = Square()
    s2.set_x(10)

    with tempconfig({"dry_run": True, "quality": "low_quality"}):
        scene = MovingCameraScene()
        scene.add(s1, s2)
        scene.play(scene.camera.auto_zoom([s1, s2]))

    assert scene.camera.frame_width == abs(
        s1.get_left()[0] - s2.get_right()[0],
    ) and scene.camera.frame.get_center()[0] == (
        abs(s1.get_center()[0] + s2.get_center()[0]) / 2
    )
