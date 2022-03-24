from __future__ import annotations

from manim import pluggy_hookimpl


@pluggy_hookimpl
def default_font() -> str:
    # Returning empty string, just uses the default font.
    return ""
