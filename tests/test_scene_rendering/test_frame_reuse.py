"""Renderers draw again only what changed since the previous frame.

The drawing of the mobjects at the start of the draw order that did not change is
copied from the previous frame (see :mod:`manim.renderer._reuse`). A frame must
look the same as one drawn from scratch, whatever changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from manim import (
    DOWN,
    GREEN,
    LEFT,
    OUT,
    PURE_BLUE,
    RED,
    RIGHT,
    UP,
    Circle,
    Dot,
    MovingCameraScene,
    Square,
    ThreeDScene,
    tempconfig,
)

RENDERERS = pytest.mark.parametrize("renderer", ["cairo", "opengl"])


def _settings(renderer):
    return tempconfig(
        {
            "renderer": renderer,
            "dry_run": True,
            "pixel_width": 192,
            "pixel_height": 108,
        }
    )


def _frame(scene):
    """Draw a frame of the scene as a render does, and return its pixels."""
    return scene.renderer.render(*scene._get_manager()._frame()).pixels()


def _build():
    back = Square(3, fill_opacity=1, color=RED).shift(LEFT)
    front = Circle(fill_opacity=0.8, color=GREEN)
    return back, front


CHANGES = {
    "color": lambda back, front, scene: back.set_fill(PURE_BLUE),
    "points": lambda back, front, scene: back.shift(DOWN + RIGHT),
    "order": lambda back, front, scene: scene.bring_to_front(back),
    "camera": lambda back, front, scene: scene.camera.frame.shift(LEFT),
    "background": lambda back, front, scene: setattr(
        scene.camera, "background_color", PURE_BLUE
    ),
    "removed": lambda back, front, scene: scene.remove(front),
    "z_index": lambda back, front, scene: back.set_z_index(1),
}


@RENDERERS
@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
def test_frames_after_a_change_look_as_drawn_from_scratch(renderer, change):
    with _settings(renderer):
        scene = MovingCameraScene()
        back, front = _build()
        moving = Dot().shift(UP)
        scene.add(back, front, moving)
        # Three frames in which only the dot moves: the third copies the drawing
        # of the square and the circle.
        for _ in range(3):
            moving.shift(0.1 * RIGHT)
            _frame(scene)
        change(back, front, scene)
        reused = _frame(scene)
        scene.renderer.close()

        fresh_scene = MovingCameraScene()
        back, front = _build()
        fresh_scene.add(back, front, Dot().shift(UP + 0.3 * RIGHT))
        change(back, front, fresh_scene)
        fresh = _frame(fresh_scene)
        fresh_scene.renderer.close()

    np.testing.assert_array_equal(reused, fresh)


@RENDERERS
def test_unchanged_mobjects_are_not_drawn_again(renderer, monkeypatch):
    with _settings(renderer):
        scene = MovingCameraScene()
        still = [Square(0.5).shift(x * RIGHT) for x in range(-3, 3)]
        moving = Dot()
        scene.add(*still, moving)
        drawn = _count_drawn_paths(monkeypatch, scene)
        for _ in range(4):
            moving.shift(0.1 * UP)
            _frame(scene)
        scene.renderer.close()

    # The first two frames draw everything; then only the dot is drawn.
    assert drawn[:2] == [7, 7]
    assert drawn[2:] == [1, 1]


@RENDERERS
def test_reuse_follows_the_depth_order_of_3d_scenes(renderer):
    def build(scene):
        near = Square(fill_opacity=1, color=RED).shift(OUT)
        far = Square(fill_opacity=1, color=GREEN).shift(0.5 * RIGHT)
        for square in (near, far):
            square.set_shade_in_3d(True)
        scene.set_camera_orientation(phi=1.0, theta=-1.2)
        scene.add(near, far)
        return near, far

    with _settings(renderer):
        scene = ThreeDScene()
        near, far = build(scene)
        for _ in range(3):
            _frame(scene)
        # Moving the far square in front changes the order of both.
        far.shift(2 * OUT)
        reused = _frame(scene)
        scene.renderer.close()

        fresh_scene = ThreeDScene()
        near, far = build(fresh_scene)
        far.shift(2 * OUT)
        fresh = _frame(fresh_scene)
        fresh_scene.renderer.close()

    np.testing.assert_array_equal(reused, fresh)


@RENDERERS
def test_mobjects_drawn_from_other_state_are_drawn_again(renderer):
    from manim.mobject.types.image_mobject import AbstractImageMobject

    class Flashing(AbstractImageMobject):
        """An image whose pixels come from an attribute, not a field."""

        _drawn_from_fields = False

        def __init__(self):
            self.channel = 0
            super().__init__(scale_to_resolution=4)

        def get_pixel_array(self):
            pixels = np.zeros((4, 4, 4), dtype=np.uint8)
            pixels[..., self.channel] = pixels[..., 3] = 255
            return pixels

    with _settings(renderer):
        scene = MovingCameraScene()
        image = Flashing().scale(2)
        moving = Dot().shift(UP)
        scene.add(image, moving)
        for _ in range(3):
            moving.shift(0.1 * RIGHT)
            _frame(scene)
        image.channel = 2
        frame = _frame(scene)
        scene.renderer.close()

    height, width, _ = frame.shape
    np.testing.assert_array_equal(frame[height // 2, width // 2, :3], [0, 0, 255])


def _count_drawn_paths(monkeypatch, scene):
    """Patch the renderer's path drawing; return the list of paths drawn per frame."""
    drawn: list[int] = []
    if scene.renderer.__class__.__name__ == "CairoRenderer":
        from manim.renderer.cairo.rendering import _CairoDrawingContext

        original = _CairoDrawingContext.draw_nodes

        def draw_nodes(self, nodes):
            if self.target is scene.renderer._target:
                drawn[-1] += len(nodes)
            return original(self, nodes)

        monkeypatch.setattr(_CairoDrawingContext, "draw_nodes", draw_nodes)
    else:
        from manim.renderer.opengl._drawing import SnapshotDrawing

        original = SnapshotDrawing._draw_nodes

        def draw_nodes(self, nodes, target, views):
            drawn[-1] += len(nodes)
            return original(self, nodes, target, views)

        monkeypatch.setattr(SnapshotDrawing, "_draw_nodes", draw_nodes)

    render = scene.renderer.render

    def counting_render(nodes, config):
        drawn.append(0)
        return render(nodes, config)

    monkeypatch.setattr(scene.renderer, "render", counting_render)
    return drawn
