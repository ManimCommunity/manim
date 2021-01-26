import pkg_resources
from importlib import import_module
from .. import config, logger

__all__ = []

# don't know what is better to do here!
module_type = type(pkg_resources)
function_type = type(import_module)

plugins_requested: list = config["plugins"]

for plugin in pkg_resources.iter_entry_points("manim.plugins"):
    if plugin.name in plugins_requested:
        loaded_plugin = plugin.load()
        if isinstance(loaded_plugin, module_type):
            # it is a module so it can't be called
            # see if __all__ is defined
            # if it is defined use that to load all the modules necessary
            # essentially this would be similar to `from plugin import *``
            # if not just import the module with the plugin name
            if hasattr(loaded_plugin, "__all__"):
                for thing in loaded_plugin.__all__:
                    exec(f"{thing}=loaded_plugin.{thing}")
                    __all__.append(thing)
            else:
                exec(plugin.name + "=loaded_plugin")
                __all__.append(plugin.name)
        elif isinstance(loaded_plugin, function_type):
            # call the function first
            # it will return a list of modules to add globally
            # finally add it
            lists = loaded_plugin()
            for l in lists:
                exec(l.__name__ + "=l")
                __all__.append(l.__name__)
        plugins_requested.remove(plugin.name)
else:
    if plugins_requested != []:
        logger.warning("Missing Plugins: %s", plugins_requested)
