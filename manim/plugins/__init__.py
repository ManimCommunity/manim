from __future__ import annotations

from .. import config, logger
from .plugins_flags import get_plugins, list_plugins

__all__ = [
    "get_plugins",
    "list_plugins",
]

requested_plugins: set[str] = set(config["plugins"])

if not requested_plugins.issubset(get_plugins().keys()):
    logger.warning("Missing Plugins: %s", requested_plugins)
