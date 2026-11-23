"""Renderers draw what a FrameBuilder makes from a scene, and nothing else."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from manim import (
    BLUE,
    PURE_RED,
    RIGHT,
    Camera,
    ImageMobjectFromCamera,
    MultiCamera,
    Square,
    tempconfig,
)
from manim.renderer.cairo import CairoRenderer
from manim.renderer.frame import FrameBuilder
from manim.renderer.opengl import OpenGLRenderer
from manim.renderer.protocol import CameraView, RenderConfig


def _scene_contents():
    camera = MultiCamera(background_color=BLUE)
    view = ImageMobjectFromCamera(Camera(background_color=PURE_RED)).set(height=2)
    view.shift(3 * RIGHT)
    camera.add_image_mobject_from_camera(view)
    return camera, [Square(fill_opacity=1), view]


def test_frames_carry_the_camera_and_its_views():
    camera, mobjects = _scene_contents()
    nodes, config = FrameBuilder(160, 90).build(camera, mobjects)

    assert isinstance(config, RenderConfig)
    assert (config.pixel_width, config.pixel_height) == (160, 90)
    assert config.background_color == camera.background_color
    np.testing.assert_allclose(config.camera.center, camera.frame_center)
    assert [node.uid for node in nodes] == [m.uid for m in mobjects]
    (view,) = config.camera_views
    assert isinstance(view, CameraView)
    assert view.uid == mobjects[1].uid
    # A view does not show itself.
    assert [node.uid for node in view.nodes] == [mobjects[0].uid]
    assert view.config.pixel_height == int(90 * 2 / camera.frame_height)


@pytest.mark.parametrize("renderer_class", [CairoRenderer, OpenGLRenderer])
def test_renderers_draw_frames_without_scenes(renderer_class):
    with tempconfig(
        {"dry_run": True, "pixel_width": 160, "pixel_height": 90, "format": "none"}
    ):
        renderer = renderer_class()
        camera, mobjects = _scene_contents()
        try:
            pixels = renderer.render(
                *FrameBuilder(160, 90).build(camera, mobjects)
            ).pixels()
        finally:
            renderer.close()

    assert pixels.shape == (90, 160, 4)
    red = (pixels[..., 0] > 200) & (pixels[..., 2] < 50)
    blue_background = pixels[2, 2, 2] > 200
    assert red[:, 100:].any()  # the view shows its camera's background
    assert blue_background


def test_frames_and_their_views_refer_to_the_scene_resolution():
    # Widths given in pixels (point clouds, mesh seams) are meant at the size of
    # the scene's frames, also in camera views, which have fewer pixels.
    camera, mobjects = _scene_contents()
    _, config = FrameBuilder(160, 90).build(camera, mobjects)
    (view,) = config.camera_views
    assert config.reference_pixel_size == (160, 90)
    assert view.config.reference_pixel_size == (160, 90)
    assert (view.config.pixel_width, view.config.pixel_height) != (160, 90)


@pytest.mark.parametrize("renderer", ["cairo", "opengl"])
def test_frames_are_drawn_at_the_size_saved_with_the_scene(renderer):
    # The Manager takes the frame size from the scene's session, not from the
    # renderer or the configuration when the frame is drawn.
    with tempconfig(
        {
            "renderer": renderer,
            "dry_run": True,
            "pixel_width": 160,
            "pixel_height": 90,
            "format": "none",
        }
    ):
        from manim import Scene

        scene = Scene()
    with (
        tempconfig({"pixel_width": 64, "pixel_height": 64}),
        scene._get_manager() as manager,
    ):
        nodes, config = manager._frame()
        pixels = scene.renderer.render(nodes, config).pixels()
    assert scene.session_spec.pixel_size == (160, 90)
    assert (config.pixel_width, config.pixel_height) == (160, 90)
    assert pixels.shape == (90, 160, 4)


@pytest.mark.parametrize("renderer_class", [CairoRenderer, OpenGLRenderer])
def test_point_thickness_follows_the_reference_resolution(renderer_class):
    # The renderer reads the scale of point clouds from the frame, not from the
    # global configuration.
    from manim import PMobject

    def drawn(reference_width):
        with tempconfig({"dry_run": True, "pixel_width": 160, "pixel_height": 90}):
            renderer = renderer_class()
            cloud = PMobject(stroke_width=8)
            cloud.add_points(np.array([[0.0, 0.0, 0.0]]))
            builder = FrameBuilder(160, 90)
            nodes, config = builder.build(Camera(), [cloud])
            reference = (reference_width, reference_width * 90 // 160)
            config = dataclasses.replace(config, reference_pixel_size=reference)
            try:
                pixels = renderer.render(nodes, config).pixels()
            finally:
                renderer.close()
        return int((pixels[..., :3].max(axis=-1) > 0).sum())

    # The width is meant at the reference size: drawn into a frame a quarter as
    # wide, a point is about four times as wide (as camera views draw it).
    assert drawn(640) > 4 * drawn(160) > 0


def test_mobjects_fixed_in_frame_come_after_all_others():
    # Fixed mobjects have no place in the scene's depth: drawn last, in scene order
    # and by z-index among themselves (R24).
    label = Square().fix_in_frame().set_z_index(-1)
    other_label = Square().fix_in_frame()
    shape, raised = Square(), Square().set_z_index(1)
    nodes, _ = FrameBuilder(16, 9).build(Camera(), [label, shape, other_label, raised])
    assert [node.uid for node in nodes] == [
        shape.uid,
        raised.uid,
        label.uid,
        other_label.uid,
    ]


@pytest.mark.parametrize("renderer", ["cairo", "opengl"])
def test_a_label_fixed_in_frame_is_drawn_over_a_later_path(renderer):
    from manim import PURE_GREEN, Scene

    with tempconfig(
        {"renderer": renderer, "dry_run": True, "pixel_width": 64, "pixel_height": 36}
    ):
        scene = Scene()
        label = Square(2, fill_color=PURE_RED, fill_opacity=1, stroke_width=0)
        label.fix_in_frame().set_z_index(-1)
        cover = Square(4, fill_color=PURE_GREEN, fill_opacity=1, stroke_width=0)
        scene.add(label, cover)
        try:
            pixel = np.asarray(scene.get_image())[18, 32]
        finally:
            scene.renderer.close()
    assert pixel[0] > 200
    assert pixel[1] < 50


def test_view_keys_differ_in_every_camera_state_field_and_light():
    from manim.renderer._reuse import view_key
    from manim.renderer.protocol import LightState

    light = LightState(np.zeros(3), (1.0, 1.0, 1.0), 10.0)
    ambient = (0.1, 0.1, 0.1)
    state = Camera()._state()

    def key(state=state, reference=(16, 9), lights=(light,), ambient=ambient):
        return view_key(state, (16, 9), reference, None, lights, ambient)

    base = key()
    changes = {
        "center": np.ones(3),
        "frame_width": 1.0,
        "frame_height": 1.0,
        "rotation": -np.identity(3),
        "focal_distance": 3.0,
        "zoom": 2.0,
        "default_frame_width": 1.0,
    }
    assert changes.keys() == {field.name for field in dataclasses.fields(state)}
    for name, value in changes.items():
        changed = dataclasses.replace(state, **{name: value})
        assert key(state=changed) != base, name
    assert key(reference=(32, 18)) != base
    assert key(lights=()) != base
    assert key(lights=(light, light)) != base
    for name, value in {
        "position": np.ones(3),
        "rgb": (1.0, 0.0, 0.0),
        "intensity": 20.0,
    }.items():
        changed = dataclasses.replace(light, **{name: value})
        assert key(lights=(changed,)) != base, name
    assert key(ambient=(0.2, 0.1, 0.1)) != base
    assert key(lights=(LightState(np.zeros(3), (1.0, 1.0, 1.0), 10.0),)) == base


@pytest.mark.parametrize("renderer", ["cairo", "opengl"])
def test_a_fixed_label_is_not_ordered_by_the_depth_of_the_scene(renderer):
    # Under a 3D camera, paths with a depth reference are drawn far to near before
    # those without; a fixed label is drawn last even if it has one.
    from manim import PURE_GREEN, ThreeDScene

    with tempconfig(
        {"renderer": renderer, "dry_run": True, "pixel_width": 64, "pixel_height": 36}
    ):
        scene = ThreeDScene()
        label = Square(2, fill_color=PURE_RED, fill_opacity=1, stroke_width=0)
        label.set_shade_in_3d(True).fix_in_frame()
        cover = Square(4, fill_color=PURE_GREEN, fill_opacity=1, stroke_width=0)
        scene.add(label, cover)
        try:
            pixel = np.asarray(scene.get_image())[18, 32]
        finally:
            scene.renderer.close()
    assert pixel[0] > pixel[1]
