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

    value: Any
    if name == "CairoRenderer":
        from .renderer import CairoRenderer

        value = CairoRenderer
    else:
        from .camera import Camera, MovingCamera, MultiCamera, ThreeDCamera

        camera_classes: dict[str, Any] = {
            "Camera": Camera,
            "MovingCamera": MovingCamera,
            "MultiCamera": MultiCamera,
            "ThreeDCamera": ThreeDCamera,
        }
        value = camera_classes[name]

    globals()[name] = value
    return value
