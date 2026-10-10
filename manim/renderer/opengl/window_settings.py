"""Saved configuration for an OpenGL preview window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..._config.utils import ManimConfig


@dataclass(frozen=True)
class _WindowSettings:
    """Values needed to size and place one native preview window.

    These values are copied from configuration. Window creation uses them when
    selecting a monitor and setting the window's size and position. The native
    backend may use different framebuffer pixel dimensions on HiDPI displays.
    """

    size: str | tuple[int, ...]
    monitor: int
    fullscreen: bool
    frame_width: float
    frame_height: float
    position: str

    @classmethod
    def from_config(cls, config: ManimConfig) -> _WindowSettings:
        return cls(
            size=config.window_size,
            monitor=config.window_monitor,
            fullscreen=config.fullscreen,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
            position=config.window_position,
        )
