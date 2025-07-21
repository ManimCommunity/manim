"""Module operations are functions that help to create runtime python modules"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import sys
import types
import warnings
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def module_from_text(code: str) -> types.ModuleType:
    """Creates a input prompt in which user can insert a code that will be asserted and executed."""
    module = types.ModuleType("RuntimeTEXT")
    try:
        # NOTE Code executer: is needed to resolve imports and other code
        exec(code, module.__dict__)
        return module
    except Exception as e:
        raise RuntimeError(f"Could not parse code from text: {e}") from e


def module_from_file(file_path: Path) -> types.ModuleType:
    """Resolve a Python module  from python file.

    Parameters
    ----------

    file_path
        location of file as path-object
    """
    if not file_path.exists() and file_path.suffix == ".py":
        raise ValueError(f"{file_path} is not a valid python script.")

    module_name = "runtimeFile" + ".".join(file_path.with_suffix("").parts)

    warnings.filterwarnings("default", category=DeprecationWarning, module=module_name)
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if isinstance(spec, importlib.machinery.ModuleSpec):
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            sys.path.insert(0, str(file_path.parent.absolute()))
            spec.loader.exec_module(module)
        else:
            raise ValueError("Failed to create ModuleSpec")

    except Exception as e:
        raise RuntimeError("Module creation from file failed") from e
    else:
        return module


def search_classes_from_module(
    module: types.ModuleType, class_type: type[T]
) -> list[type[T]]:
    """Search and return all occurrence of specified type classes.

    Parameters
    -----------
    module
        Module object
    class_type:
        Type of searched classes
    """

    def is_child_scene(obj: Any) -> bool:
        return (
            isinstance(obj, type)
            and issubclass(obj, class_type)
            and obj != class_type
            and obj.__module__.startswith(module.__name__)
        )

    classes = [member[1] for member in inspect.getmembers(module, is_child_scene)]

    if len(classes) == 0:
        raise ValueError(f"Could not found any classes of type {class_type.__name__}")
    return classes


def scene_classes_for_gui(path: str, class_type: type[T]) -> list[type[T]]:
    """Specified interface of  dearpyGUI to fetch Scene-class instances"""
    module = module_from_file(Path(path))
    return search_classes_from_module(module, class_type)
