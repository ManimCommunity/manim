import glob
import pkgutil
import importlib

from os.path import dirname, basename, isfile, join

# Imports all files directly inside of plugins folder
modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")]
plugin_module = importlib.import_module("plugin_name.package", ".")