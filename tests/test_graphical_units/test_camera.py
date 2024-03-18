from __future__ import annotations

from manim import *
from manim.utils.testing.frames_comparison import frames_comparison

__module_test__ = "camera"


@frames_comparison(last_frame=False, base_scene=ThreeDScene)
def test_MoveCameraX(scene):
    square = Square()
    scene.add(square)

    image_mobject = ImageMobject(
        np.uint8([[255 for x in range(50)] for y in range(50)])
    )
    scene.add(image_mobject)

    scene.move_camera(frame_center=[3.0, 0.0, 0])


@frames_comparison(last_frame=False, base_scene=ThreeDScene)
def test_MoveCameraY(scene):
    square = Square()
    scene.add(square)

    image_mobject = ImageMobject(
        np.uint8([[255 for x in range(50)] for y in range(50)])
    )
    scene.add(image_mobject)

    scene.move_camera(frame_center=[0.0, 3.0, 0])
