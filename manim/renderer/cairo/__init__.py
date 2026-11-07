"""Cairo rendering backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .camera import Camera, MovingCamera, MultiCamera, ThreeDCamera
    from .renderer import CairoRenderer

__all__ = [
    "Camera",
    "CairoRenderer",
    "MovingCamera",
    "MultiCamera",
    "ThreeDCamera",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Import on first use: the renderer and cameras import much of manim.
    import importlib

    module = importlib.import_module(
        ".renderer" if name == "CairoRenderer" else ".camera", __name__
    )
    value: Any = getattr(module, name)

    globals()[name] = value
    return value
