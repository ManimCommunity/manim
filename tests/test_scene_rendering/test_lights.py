"""Lights belong to the scene: they move, animate and turn off like mobjects."""

from __future__ import annotations

import numpy as np
import pytest

from manim import (
    DEGREES,
    LEFT,
    RIGHT,
    UP,
    FadeOut,
    Manager,
    PointLight,
    Scene,
    Sphere,
    Square,
    ThreeDScene,
    Transform,
    tempconfig,
)
from manim.light import DEFAULT_LIGHT_INTENSITY

RENDERERS = pytest.mark.parametrize("renderer", ["cairo", "opengl"])


def _settings(renderer, **more):
    return tempconfig(
        {
            "renderer": renderer,
            "dry_run": True,
            "pixel_width": 96,
            "pixel_height": 54,
            **more,
        }
    )


def _lit_scene(scene):
    scene.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)
    scene.add(Sphere(resolution=(16, 8)))


def _image(scene):
    return np.asarray(scene.get_image()).astype(int)


def test_a_light_moves_and_animates_like_a_mobject():
    light = PointLight(LEFT)
    light.generate_target()
    light.target.move_to(RIGHT).set_intensity(10)
    animation = Transform(light, light.target)
    animation.begin()
    animation.interpolate(0.5)
    np.testing.assert_allclose(light.position, [0, 0, 0])
    assert light.intensity == pytest.approx((DEFAULT_LIGHT_INTENSITY + 10) / 2)
    animation.interpolate(1)
    assert light.intensity == 10


def test_fading_out_a_light_turns_it_off():
    with tempconfig({"dry_run": True}):
        scene = ThreeDScene()
        light = PointLight(UP)
        scene.add(light)
        animation = FadeOut(light)
        animation.begin()
        animation.interpolate(0.5)
        assert 0 < light.intensity < DEFAULT_LIGHT_INTENSITY
        animation.interpolate(1)
        assert light.intensity == 0
        animation.clean_up_from_scene(scene)
        assert light not in scene.mobjects


def test_the_default_light_survives_clearing_the_scene():
    with tempconfig({"dry_run": True}):
        scene = ThreeDScene()
        scene.add(Square())
        light = scene.light
        scene.clear()
        assert scene.light is light
        assert light not in scene.mobjects


@RENDERERS
def test_a_moving_light_changes_a_lit_mesh_frame_by_frame(renderer):
    with _settings(renderer):
        scene = ThreeDScene()
        _lit_scene(scene)
        try:
            frames = []
            for x in (-6, 0, 6):
                scene.light.move_to([x, -9, 10])
                frames.append(_image(scene))
        finally:
            scene.renderer.close()
    assert not np.array_equal(frames[0], frames[1])
    assert not np.array_equal(frames[1], frames[2])


@RENDERERS
def test_without_lights_a_lit_mesh_shows_its_base_colour(renderer):
    # "No light, no shading": a lit sphere without lights looks as an unlit one.
    from manim.material import UnlitMaterial

    def image(light, material=None):
        scene = ThreeDScene()
        scene.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)
        sphere = Sphere(resolution=(16, 8))
        if material is not None:
            sphere.material = material
        scene.add(sphere)
        if not light:
            scene.light = None
        try:
            return _image(scene)
        finally:
            scene.renderer.close()

    with _settings(renderer):
        lit = image(light=True)
        dark = image(light=False)
        unlit = image(light=True, material=UnlitMaterial())
    assert not np.array_equal(dark, lit)
    np.testing.assert_array_equal(dark, unlit)


@RENDERERS
def test_lights_added_to_the_scene_light_it(renderer):
    with _settings(renderer):
        scene = Scene()
        scene.add(Sphere(resolution=(16, 8)))
        try:
            before = _image(scene)
            scene.add(PointLight([0, 0, 6]))
            after = _image(scene)
        finally:
            scene.renderer.close()
    assert not np.array_equal(before, after)


def _cache_hits(tmp_path, change):
    class Lit(ThreeDScene):
        def construct(self):
            change(self)
            self.add(Sphere(resolution=(8, 4)))
            self.play(self.camera.frame.animate.shift(0.1 * RIGHT), run_time=0.25)

    scene = Lit()
    writer = scene._get_manager().file_writer
    hits = []
    lookup = writer.is_already_cached
    writer.is_already_cached = lambda key: hits.append(lookup(key)) or hits[-1]
    Manager(scene).render() if scene.manager is None else scene.render()
    return hits


@pytest.mark.parametrize(
    "change",
    [
        lambda scene: setattr(scene, "ambient_light", 0.5),
        lambda scene: scene.light.move_to(3 * LEFT),
        lambda scene: scene.light.set_intensity(10),
    ],
    ids=["ambient light", "light position", "light intensity"],
)
def test_cached_plays_are_not_reused_after_a_lighting_change(tmp_path, change):
    settings = {
        "format": "mp4",
        "media_dir": str(tmp_path),
        "frame_rate": 4,
        "pixel_width": 32,
        "pixel_height": 18,
        "disable_caching": False,
        "progress_bar": "none",
    }
    with tempconfig(settings):
        assert _cache_hits(tmp_path, lambda scene: None) == [False]
        assert _cache_hits(tmp_path, lambda scene: None) == [True]
        assert _cache_hits(tmp_path, change) == [False]


def test_animating_the_scene_light_keeps_it_apart_from_the_mobjects():
    class Moves(ThreeDScene):
        def construct(self):
            self.add(Sphere(resolution=(8, 4)))
            self.play(self.light.animate.move_to(3 * LEFT), run_time=0.25)

    with tempconfig({"dry_run": True, "frame_rate": 4, "progress_bar": "none"}):
        scene = Moves()
        scene.render()
    np.testing.assert_allclose(scene.light.position, 3 * LEFT)
    assert scene.light not in scene.mobjects
