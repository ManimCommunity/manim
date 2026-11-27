"""Scene configuration under the OpenGL renderer.

Ported from tests/opengl, which tested the OpenGL mobject classes; these are the
tests of that directory that check the renderer rather than the mobjects.
"""

from __future__ import annotations

import numpy as np

from manim import WHITE, Scene, Square, VMobject


class _SquareScene(Scene):
    def construct(self):
        self.add(Square())
        self.wait(1)


def test_background_color(config, using_opengl_renderer, dry_run):
    config.background_color = WHITE
    config.verbose = "ERROR"

    scene = _SquareScene()
    with scene._get_manager():
        scene.render()
        frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [255, 255, 255, 255])


def test_dry_run_with_png_format(config, using_opengl_renderer, dry_run):
    config.disable_caching = True
    _SquareScene().render()


def test_dry_run_with_png_format_skipped_animations(
    config, using_opengl_renderer, dry_run
):
    config.format = "png"
    config.disable_caching = True
    _SquareScene(skip_animations=True).render()


def test_scene_add_remove(using_opengl_renderer, dry_run):
    scene = Scene()
    shapes = [VMobject() for _ in range(3)]
    assert scene.add(*shapes) is scene
    # Adding a mobject again moves it to the front, without a second copy.
    scene.add(shapes[0])
    assert scene.mobjects == [shapes[1], shapes[2], shapes[0]]
    assert scene.remove(shapes[1]) is scene
    assert scene.mobjects == [shapes[2], shapes[0]]
