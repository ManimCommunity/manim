from __future__ import annotations

from manim import MovingCamera, Square


def test_movingcamera_auto_zoom():
    camera = MovingCamera()
    square = Square()
    margin = 0.5
    camera.auto_zoom([square], margin=margin, animate=False)
    assert camera.frame.height == square.height + margin
