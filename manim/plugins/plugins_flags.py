"""Plugin Managing Utility"""

from __future__ import annotations

import sys
from typing import Any

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from manim._config import console

__all__ = ["list_plugins"]


def get_plugins() -> dict[str, Any]:
    plugins: dict[str, Any] = {
        entry_point.name: entry_point.load()
        for entry_point in entry_points(group="manim.plugins")
    }
    return plugins


def list_plugins() -> None:
    console.print("[green bold]Plugins:[/green bold]", justify="left")

    plugins = get_plugins()
    for plugin_name in plugins:
        console.print(f" • {plugin_name}")
