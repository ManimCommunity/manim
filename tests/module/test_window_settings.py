"""Window input resolution without acquiring native display resources."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from manim import config, tempconfig
from manim.renderer.opengl.window_settings import _WindowSettings


class _StopBeforeNativeOpen(Exception):
    pass


@pytest.fixture
def window_attempt(monkeypatch):
    from manim.renderer.opengl import window as window_module

    attempts = []

    def stop(window, **kwargs):
        attempts.append(kwargs)
        raise _StopBeforeNativeOpen

    monkeypatch.setattr(window_module.PygletWindow, "__init__", stop)
    monkeypatch.setattr(
        window_module,
        "get_monitors",
        lambda: [
            SimpleNamespace(width=1200, height=800, x=0, y=0),
            SimpleNamespace(width=1600, height=1000, x=1200, y=0),
        ],
    )

    def attempt(**kwargs):
        with pytest.raises(_StopBeforeNativeOpen):
            window_module.Window(Mock(), **kwargs)
        return attempts[-1]

    return attempt


def test_omitted_window_size_uses_each_constructions_config(window_attempt):
    # The module has already been imported by the fixture.
    with tempconfig({"window_size": (240, 180)}):
        assert window_attempt()["size"] == (240, 180)
    with tempconfig({"window_size": (360, 240)}):
        assert window_attempt()["size"] == (360, 240)


def test_explicit_window_size_overrides_config(window_attempt):
    with tempconfig({"window_size": (240, 180)}):
        assert window_attempt(window_size=(320, 200))["size"] == (320, 200)


def test_supplied_settings_do_not_reread_size(window_attempt):
    with tempconfig({"window_size": (240, 180)}):
        settings = _WindowSettings.from_config(config)
    with tempconfig({"window_size": (360, 240)}):
        assert window_attempt(_settings=settings)["size"] == (240, 180)
        assert window_attempt(_settings=settings, window_size="320,200")["size"] == (
            320,
            200,
        )


@pytest.mark.parametrize(
    ("fullscreen", "expected"), [(False, (800, 450)), (True, (1600, 900))]
)
def test_default_size_uses_captured_monitor_and_geometry(
    window_attempt, fullscreen, expected
):
    with tempconfig(
        {
            "window_size": "default",
            "window_monitor": 1,
            "fullscreen": fullscreen,
            "frame_width": 16,
            "frame_height": 9,
        }
    ):
        settings = _WindowSettings.from_config(config)
    with tempconfig(
        {
            "window_monitor": 0,
            "fullscreen": not fullscreen,
            "frame_width": 8,
            "frame_height": 8,
        }
    ):
        assert window_attempt(_settings=settings)["size"] == expected


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("UL", (10, -20)),
        ("DR", (1010, 680)),
        ("42,17", (17, 42)),
    ],
)
def test_initial_position_uses_captured_setting(position, expected):
    from manim.renderer.opengl.window import Window

    with tempconfig({"window_position": position}):
        settings = _WindowSettings.from_config(config)
    window = Window.__new__(Window)
    window._settings = settings
    monitor = SimpleNamespace(width=1200, height=800, x=10, y=20)
    with tempconfig({"window_position": "ORIGIN"}):
        assert window.find_initial_position((200, 100), monitor) == expected


def test_capture_does_not_query_monitors(monkeypatch):
    from manim.renderer.opengl import window as window_module

    query = Mock(side_effect=AssertionError("capture must not query the display"))
    monkeypatch.setattr(window_module, "get_monitors", query)
    _WindowSettings.from_config(config)
    query.assert_not_called()
