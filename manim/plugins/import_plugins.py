from __future__ import annotations

import types

import pkg_resources

from .. import config, logger

__all__ = []


plugins_requested: list[str] = config["plugins"]
if "" in plugins_requested:
    plugins_requested.remove("")
for plugin in pkg_resources.iter_entry_points("manim.plugins"):
    if plugin.name not in plugins_requested:
        continue
    loaded_plugin = plugin.load()
    if isinstance(loaded_plugin, types.ModuleType):
        # it is a module so it can't be called
        # see if __all__ is defined
        # if it is defined use that to load all the modules necessary
        # essentially this would be similar to `from plugin import *``
        # if not just import the module with the plugin name
        if hasattr(loaded_plugin, "__all__"):
            for thing in loaded_plugin.__all__:  # type: ignore
                exec(f"{thing}=loaded_plugin.{thing}")
                __all__.append(thing)
        else:
            exec(f"{plugin.name}=loaded_plugin")
            __all__.append(plugin.name)
    elif isinstance(loaded_plugin, types.FunctionType):
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
