import importlib.util
import sys
from types import ModuleType


# TODO: Replace with proper lazy imports when switching to 3.15
def lazy_import(name: str, package: str | None = None) -> ModuleType:
    specification = importlib.util.find_spec(name, package)
    if specification is None:
        raise ValueError("Couldn't find the module specification.")
    if specification.loader is None:
        raise ValueError("Module doesn't have a loader")
    lazy_loader = importlib.util.LazyLoader(specification.loader)
    specification.loader = lazy_loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    lazy_loader.exec_module(module)
    return module
