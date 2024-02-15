from __future__ import annotations

from manim import config, logger

from .plugins_flags import get_plugins, list_plugins

__all__ = [
    "get_plugins",
    "list_plugins",
]

requested_plugins: set[str] = set(config["plugins"])
missing_plugins = requested_plugins - set(get_plugins().keys())


if missing_plugins:
    logger.warning("Missing Plugins: %s", missing_plugins)
