"""Interactive redraws display frames without advancing animation time."""

from unittest.mock import Mock

import pytest

from manim import Scene, tempconfig


@pytest.fixture
def scene():
    with tempconfig({"renderer": "opengl", "format": "none", "live_preview": True}):
        scene = Scene()
        with scene._get_manager():
            yield scene


def test_preview_presents_without_advancing_time(scene, monkeypatch):
    manager = scene.manager
    manager.time = 2
    draw = Mock()
    monkeypatch.setattr(scene.renderer, "render", draw)
    present = Mock()
    monkeypatch.setattr(manager.preview, "present", present)
    manager._render_preview_frame(-1)
    draw.assert_called_once()
    # The frame drawn is shown, without reading its pixels.
    present.assert_called_once_with(draw.return_value)
    draw.return_value.pixels.assert_not_called()
    assert scene.time == 2


def test_embed_redraws_through_manager_before_and_after_commands(scene, monkeypatch):
    redraw = Mock()
    monkeypatch.setattr(scene.manager, "_render_preview_frame", redraw)
    shell = Mock()
    monkeypatch.setattr("IPython.terminal.embed.InteractiveShellEmbed", lambda: shell)
    with pytest.raises(Exception, match="Exiting scene"):
        scene.embed()
    redraw.assert_called_once_with(-1)
    event, callback = shell.events.register.call_args.args
    assert event == "post_run_cell"
    callback()
    assert redraw.call_count == 2
    assert scene.time == 0


def test_interactive_idle_redraw_uses_manager(scene, monkeypatch):
    monkeypatch.setattr("manim.scene.scene.Observer", Mock())

    def redraw(_):
        scene.quit_interaction = True

    redraw = Mock(side_effect=redraw)
    monkeypatch.setattr(scene.manager, "_render_preview_frame", redraw)
    scene.interact(Mock(pt_app=None), Mock())
    redraw.assert_called_once()
    assert scene.time == 0


def test_a_render_opens_the_window_before_setup(monkeypatch):
    # interactive_embed() in a construct() without a play needs the window.
    with tempconfig({"renderer": "opengl", "format": "none", "live_preview": True}):
        scene = Scene()
        shown = []
        manager = scene._get_manager()
        monkeypatch.setattr(manager.preview, "show", lambda: shown.append("show"))
        monkeypatch.setattr(scene, "setup", lambda: shown.append("setup"))
        monkeypatch.setattr(scene, "construct", lambda: shown.append("construct"))
        monkeypatch.setattr(manager, "_show_final_state", lambda: None)
        manager.render()
    assert shown == ["show", "setup", "construct"]


@pytest.mark.parametrize("use", ["evaluate", "get_image"])
def test_inspection_never_opens_a_window(use, monkeypatch, tmp_path):
    from manim.renderer.opengl import window as window_module

    monkeypatch.setattr(
        window_module,
        "Window",
        Mock(side_effect=AssertionError("inspection opened a window")),
    )
    with tempconfig(
        {
            "renderer": "opengl",
            "format": "none",
            "live_preview": True,
            "pixel_width": 32,
            "pixel_height": 18,
            "media_dir": str(tmp_path),
        }
    ):
        scene = Scene()
        if use == "evaluate":
            scene._get_manager().evaluate()
        else:
            assert scene.get_image().size == (32, 18)
            scene.renderer.close()
    assert scene.renderer.window is None


def test_scenes_without_a_live_preview_have_no_preview_handle():
    with tempconfig({"renderer": "opengl", "format": "none", "live_preview": False}):
        scene = Scene()
    assert scene._get_manager().preview is None
    assert scene.renderer.preview is None


def test_a_preview_shows_only_the_last_frame_of_its_renderer(monkeypatch):
    with tempconfig(
        {"renderer": "opengl", "format": "none", "live_preview": True, "dry_run": False}
    ):
        scene = Scene()
        renderer = scene.renderer
        monkeypatch.setattr(renderer, "open", lambda: None)
        monkeypatch.setattr(renderer, "_draw", lambda nodes, config: None)
        window = Mock()
        monkeypatch.setattr(renderer, "window", window)
        renderer._frame_buffer_object = Mock()
        manager = scene._get_manager()
        first = renderer.render(*manager._frame())
        second = renderer.render(*manager._frame())
        with pytest.raises(RuntimeError, match="another frame"):
            manager.preview.present(first)
        manager.preview.present(second)
        window.show.assert_called_once_with(renderer._frame_buffer_object)
        renderer._frame_buffer_object = None


def test_interaction_ends_and_closes_the_renderer_when_the_window_closes(
    scene, monkeypatch
):
    monkeypatch.setattr("manim.scene.scene.Observer", Mock())
    redraw = Mock()
    monkeypatch.setattr(scene.manager, "_render_preview_frame", redraw)
    closing = Mock()
    closing.is_closing = True
    monkeypatch.setattr(type(scene.manager), "preview", property(lambda m: closing))
    scene.interact(Mock(pt_app=None), Mock())
    redraw.assert_not_called()
    assert not scene.renderer.state.accepts_work
