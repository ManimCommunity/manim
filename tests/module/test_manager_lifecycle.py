"""3B1 acceptance probes; strict xfails identify unfinished ownership boundaries.

These are desired resource guarantees, not a specification to preserve the current
cleanup gaps. Keep legacy timed execution unchanged while making these pass.
"""

from unittest.mock import Mock

import pytest

from manim import CairoRenderer, Manager, Scene
from manim.renderer.cairo import renderer as cairo_module
from manim.scene.scene_file_writer import SceneFileWriter


@pytest.mark.xfail(
    strict=True, reason="3B1: Cairo renderer construction still allocates targets"
)
def test_cairo_renderer_shell_does_not_allocate_targets(monkeypatch):
    target = Mock(wraps=cairo_module._CairoRenderTarget)
    monkeypatch.setattr(cairo_module, "_CairoRenderTarget", target)
    renderer = CairoRenderer()
    try:
        target.assert_not_called()
    finally:
        renderer.close()


@pytest.mark.xfail(strict=True, reason="3B1: Scene construction still creates a writer")
def test_scene_construction_does_not_create_writer(dry_run, monkeypatch):
    calls = []
    original = SceneFileWriter.__init__

    def init(writer, *args, **kwargs):
        calls.append(writer)
        original(writer, *args, **kwargs)

    monkeypatch.setattr(SceneFileWriter, "__init__", init)
    scene = Scene()
    try:
        assert calls == []
    finally:
        scene.renderer.close()


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
@pytest.mark.parametrize(
    "hook",
    [
        pytest.param(
            "setup",
            marks=pytest.mark.xfail(
                strict=True, reason="3B1: setup is outside cleanup scope"
            ),
        ),
        "construct",
        pytest.param(
            "tear_down",
            marks=pytest.mark.xfail(
                strict=True, reason="3B1: teardown is outside cleanup scope"
            ),
        ),
        pytest.param(
            "post_construct",
            marks=pytest.mark.xfail(
                strict=True, reason="3B1: finalization is outside cleanup scope"
            ),
        ),
    ],
)
def test_lifecycle_failure_aborts_output_and_preserves_identity(
    dry_run, monkeypatch, hook, failure_type
):
    scene = Scene()
    manager = Manager(scene)
    failure = failure_type("user hook failed")
    abort = Mock()
    monkeypatch.setattr(scene.renderer.file_writer, "abort_encode_jobs", abort)
    monkeypatch.setattr(manager, hook, Mock(side_effect=failure))
    try:
        with pytest.raises(failure_type) as caught:
            manager.render()
        assert caught.value is failure
        abort.assert_called_once()
    finally:
        scene.renderer.close()


@pytest.mark.xfail(
    strict=True, reason="3B1: abort failure still masks the primary hook failure"
)
def test_cleanup_failure_does_not_replace_primary_exception(dry_run, monkeypatch):
    scene = Scene()
    manager = Manager(scene)
    failure = KeyboardInterrupt("construct interrupted")
    monkeypatch.setattr(manager, "construct", Mock(side_effect=failure))
    abort = Mock(side_effect=RuntimeError("cleanup failed"))
    monkeypatch.setattr(scene.renderer.file_writer, "abort_encode_jobs", abort)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            manager.render()
        assert caught.value is failure
        abort.assert_called_once()
    finally:
        scene.renderer.close()
