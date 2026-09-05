"""Shared renderer feature declarations."""

from __future__ import annotations

__all__ = ["RendererCapabilities"]

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RendererCapabilities:
    """Optional session features implemented by a renderer."""

    live_preview: bool = False
