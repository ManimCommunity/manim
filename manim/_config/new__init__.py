"""Set the global config and logger."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from .config_provider import CfgProvider
from .new import ManimConfig

# In the future, we will also read from TOMLProvider, and use the best provider depending on the existing
# configuration files.
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


@contextmanager
def tempconfig(temp: ManimConfig | dict[str, Any]) -> Iterator[None]:
    """Context manager that temporarily modifies the global ``config`` object.

    Inside the ``with`` statement, the modified config will be used.  After
    context manager exits, the config will be restored to its original state.

    Parameters
    ----------
    temp
        Object whose keys will be used to temporarily update the global
        ``config``.

    Examples
    --------

    Use ``with tempconfig({...})`` to temporarily change the default values of
    certain config options.

    .. code-block:: pycon

       >>> config["frame_height"]
       8.0
       >>> with tempconfig({"frame_height": 100.0}):
       ...     print(config["frame_height"])
       ...
       100.0
       >>> config["frame_height"]
       8.0

    """
    global config

    original = config.model_dump(exclude_unset=True)

    if isinstance(temp, ManimConfig):
        temp = temp.model_dump(exclude_unset=True)
    temp = {k: v for k, v in temp.items() if k in original}

    # In order to change the config that every module has access to,
    # we do not use assignment.  Assigning config = some_dict will just
    # make the local variable named config point to a new dictionary, it will
    # NOT change the dictionary that every module has a reference to.
    for k, v in temp:
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in original.items():
            setattr(config, k, v)
