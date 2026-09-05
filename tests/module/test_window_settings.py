"""Window input resolution without acquiring native display resources."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from manim import tempconfig


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
        lambda: [SimpleNamespace(width=1200, height=800, x=0, y=0)],
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
