from __future__ import annotations

from manim import config, logger

from .plugin_config import Hooks, plugins
from .plugins_flags import get_plugins, list_plugins

__all__ = [
    "plugins",
    "Hooks",
    "list_plugins",
    "get_plugins",
]

requested_plugins: set[str] = set(config["plugins"])
missing_plugins = requested_plugins - set(get_plugins().keys())


if missing_plugins:
    logger.warning(f"Missing Plugins: {missing_plugins}")
