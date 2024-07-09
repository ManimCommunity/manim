from __future__ import annotations

from manim import config, logger

from .plugin_config import Hooks, plugins
from .plugins_flags import get_plugins, list_plugins

__all__ = [
    "plugins",
    "Hooks",
    "get_plugins",
    "list_plugins",
]

requested_plugins: set[str] = set(config["plugins"])
missing_plugins = requested_plugins - set(get_plugins().keys())


if missing_plugins:
    logger.warning("Missing Plugins: %s", missing_plugins)
