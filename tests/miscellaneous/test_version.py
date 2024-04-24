from __future__ import annotations

from importlib.metadata import version

from manim import __name__, __version__


def test_version():
    assert __version__ == version(__name__)
