from __future__ import annotations

import numpy as np
from PIL import Image

from manim import Scene
from manim.mobject.opengl.opengl_image_mobject import OpenGLImageMobject

RED = np.array([255, 0, 0, 255])
BLUE = np.array([0, 0, 255, 255])


def _red_over_blue(height: int, width: int) -> np.ndarray:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[: height // 2, :, 0] = 255
    pixels[height // 2 :, :, 2] = 255
    return pixels


def _render_and_sample(image: OpenGLImageMobject) -> tuple[np.ndarray, np.ndarray]:
    """Render ``image`` (with height 4, i.e. half the frame height) and return
    the pixel colors in the centers of its upper and lower halves.
    """
    scene = Scene()
    scene.add(image)
    scene.renderer.update_frame(scene)
    frame = scene.renderer.get_frame()
    height, width, _ = frame.shape
    upper = frame[int(height * 3 / 8), width // 2]
    lower = frame[int(height * 5 / 8), width // 2]
    return upper, lower


def test_opengl_image_mobject_from_file(using_opengl_renderer, tmp_path):
    path = tmp_path / "red_over_blue.png"
    Image.fromarray(_red_over_blue(64, 64)).save(path)

    upper, lower = _render_and_sample(OpenGLImageMobject(path, height=4))

    np.testing.assert_array_equal(upper, RED)
    np.testing.assert_array_equal(lower, BLUE)

