"""Concrete native-window inputs, independent of resource acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..._config.utils import ManimConfig


@dataclass(frozen=True)
class _WindowSettings:
    """Values needed to size and place one native preview window.

    Capture performs no monitor queries or native imports. The caller chooses
    when to resolve these inputs; Window currently does so at construction.
    Size is a window sizing request, not an output-raster pixel contract; native
    framebuffer dimensions may differ on HiDPI displays.
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
