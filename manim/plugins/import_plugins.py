from __future__ import annotations

import sys
import types

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

from .. import config, logger

__all__ = []


plugins_requested: list[str] = config["plugins"]

if "" in plugins_requested:
    plugins_requested.remove("")

for plugin in entry_points(group="manim.plugins"):
    if plugin.name not in plugins_requested:
        continue

    loaded_plugin = plugin.load()

    if isinstance(loaded_plugin, types.ModuleType):
        # it is a module so can't be called. See if `__all__` is defined.
        # If it is, use that to load all the modules necessary.
        if hasattr(loaded_plugin, "__all__"):
            for obj in loaded_plugin.__all__:  # type: ignore
                exec(f"{obj}=loaded_plugin.{obj}")
                __all__.append(obj)
        else:
            exec(f"{plugin.name}=loaded_plugin")
            __all__.append(plugin.name)
    elif callable(loaded_plugin):
        # call the function first
        # it will return a list of modules to add globally
        # finally add it
        lists = loaded_plugin()
        for lst in lists:
            exec(f"{lst.__name__}=lst")
            __all__.append(lst.__name__)

    plugins_requested.remove(plugin.name)

if plugins_requested != []:
    logger.warning("Missing Plugins: %s", plugins_requested)
