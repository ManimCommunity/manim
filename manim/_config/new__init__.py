"""Set the global config and logger."""

from __future__ import annotations

import logging

from .config_provider import CfgProvider
from .new import ManimConfig

cfg_provider = CfgProvider()

if cfg_provider.available:
    config = cfg_provider.get_config()
else:
    config = ManimConfig()

cli_ctx_settings = config.cli_formatting.context_settings

console, error_console = config.logging.consoles
logger = config.logging.make_logger()

# TODO: temporary to have a clean terminal output when working with PIL or matplotlib
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)
