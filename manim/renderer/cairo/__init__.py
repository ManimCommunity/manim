"""Cairo rendering backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .renderer import CairoRenderer

__all__ = ["CairoRenderer"]


def __getattr__(name: str) -> Any:
    if name != "CairoRenderer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .renderer import CairoRenderer

    globals()[name] = CairoRenderer
    return CairoRenderer
