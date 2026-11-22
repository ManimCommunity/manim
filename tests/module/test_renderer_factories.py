"""Scenes select renderers with factories; any factory written against the renderer
interface works with the Manager, the play cache, images and evaluation.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from manim import Manager, Scene, Square, tempconfig
from manim.renderer.backends import Cairo, OpenGL, backend
from manim.renderer.cairo import CairoRenderer
from manim.renderer.protocol import Renderer, RendererFactory, RendererState
from tests.helpers.plain_renderer import Plain, PlainRenderer


class _Moves(Scene):
    def construct(self):
        square = Square(fill_opacity=1)
        self.add(square)
        self.play(square.animate.shift([1, 0, 0]), run_time=0.5)


def _settings(tmp_path, **more):
    # The configured renderer is another one: the factory decides.
    return tempconfig(
        {
            "renderer": "opengl",
            "format": "mp4",
            "frame_rate": 4,
            "pixel_width": 32,
            "pixel_height": 18,
            "media_dir": str(tmp_path),
            "progress_bar": "none",
            "live_preview": False,
            **more,
        }
    )


def test_the_built_in_factories_are_renderer_factories():
    for factory in (Cairo(), OpenGL(), Plain()):
        assert isinstance(factory, RendererFactory)
    assert isinstance(PlainRenderer(1), Renderer)


def test_names_select_the_built_in_factories(config):
    config.enable_wireframe = True
    assert backend("cairo", config) == Cairo()
    assert backend("OpenGL", config) == OpenGL(wireframe=True)
    config.renderer = "cairo"
    assert backend(None, config) == Cairo()
    factory = Plain(shade=3)
    assert backend(factory, config) is factory


def test_a_renderer_instance_is_rejected_with_the_factory_to_use(dry_run):
    with pytest.raises(TypeError, match="renderer factory"):
        Scene(renderer=CairoRenderer())


def test_scenes_of_one_factory_do_not_share_a_renderer(dry_run):
    factory = Plain()
    first, second = Scene(renderer=factory), Scene(renderer=factory)
    assert first.renderer is not second.renderer
    assert first.backend is second.backend is factory


def test_a_cached_render_with_a_plain_renderer(tmp_path):
    def render(factory):
        scene = _Moves(renderer=factory)
        writer = scene._get_manager().file_writer
        hits = []
        lookup = writer.is_already_cached
        writer.is_already_cached = lambda key: hits.append(lookup(key)) or hits[-1]
        scene.render()
        return scene, hits

    with _settings(tmp_path, disable_caching=False):
        scene, hits = render(Plain())
        assert hits == [False]
        assert scene._get_manager().file_writer.final_file_path.is_file()
        assert scene.renderer.drawn
        _, hits = render(Plain())
        assert hits == [True]
        # Another option value changes the pixels, so the segment is not reused.
        _, hits = render(Plain(shade=128))
        assert hits == [False]


def test_images_and_evaluation_with_a_plain_renderer(tmp_path):
    with _settings(tmp_path):
        scene = Scene(renderer=Plain(shade=200))
        scene.add(Square(fill_opacity=1))
        image = np.asarray(scene.get_image())
        assert image.shape == (18, 32, 4)
        assert image[9, 16, 0] == 200
        # The image renderer is another renderer, closed after the image.
        assert not scene.renderer.drawn

        frames = []
        evaluated = _Moves(renderer=Plain(shade=200))
        Manager(evaluated).evaluate(frame_times=[0.25], on_frame=frames.append)
        assert [frame.frame_index for frame in frames] == [1]
        assert frames[0].image.size == (32, 18)


def test_a_failed_render_closes_the_plain_renderer(tmp_path):
    class Fails(Scene):
        def construct(self):
            self.add(Square())
            self.play(Square().animate.shift([1, 0, 0]), run_time=0.5)
            raise ValueError("construct failed")

    with _settings(tmp_path):
        scene = Fails(renderer=Plain())
        with pytest.raises(ValueError, match="construct failed"):
            scene.render()
    assert scene.renderer.state is RendererState.CLOSED


def test_images_in_a_managed_scope_share_one_renderer_closed_with_the_scope(
    tmp_path, monkeypatch
):
    created = []
    factory = Plain()
    create = Plain.create

    def counting(self, preview=None):
        renderer = create(self, preview)
        created.append(renderer)
        return renderer

    monkeypatch.setattr(Plain, "create", counting)
    with _settings(tmp_path):
        scene = Scene(renderer=factory)
        scene_renderer = created.pop()
        with scene._get_manager():
            scene.get_image()
            scene.get_image()
            assert len(created) == 1
            assert created[0].state is RendererState.OPEN
        assert created[0].state is RendererState.CLOSED
        # Outside a scope, a temporary renderer is closed before returning.
        scene.get_image()
        assert len(created) == 2
        assert created[1].state is RendererState.CLOSED
    assert scene_renderer.state is RendererState.CLOSED


def test_a_failing_image_renderer_does_not_close_the_scene_renderer(tmp_path):
    with _settings(tmp_path):
        scene = Scene(renderer=Plain())
        failing = Mock(spec=PlainRenderer)
        failing.render.side_effect = RuntimeError("image failed")
        with scene._get_manager() as manager:
            manager._image_renderer = failing
            manager._image_thread = None
            with pytest.raises(RuntimeError, match="image failed"):
                scene.get_image()
            failing.close.assert_called_once_with()
            assert manager._image_renderer is None
            assert scene.renderer.state is RendererState.UNOPENED
            # The next image gets a renderer of its own.
            assert scene.get_image().size == (32, 18)


def test_a_promised_preview_without_a_handle_is_reported_by_name(tmp_path):
    from manim.renderer.protocol import RendererCapabilities

    class Promises(Plain):
        @property
        def capabilities(self):
            return RendererCapabilities(live_preview=True)

        def create(self, preview=None):
            return PlainRenderer(self.shade)

    with _settings(tmp_path, format="none", live_preview=True):
        scene = Scene(renderer=Promises())
        with pytest.raises(RuntimeError, match="PlainRenderer promised a live"):
            scene._get_manager().preview
