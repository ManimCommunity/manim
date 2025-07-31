"""Module operations are functions that help to create runtime python modules"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
import warnings
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def module_from_text(code: str) -> types.ModuleType:
    """Creates a input prompt in which user can insert a code that will be asserted and executed.

    Parameters
    ----------
    code
        code string
    """
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
        location of python file as path-object
    """
    if not file_path.exists() and file_path.suffix == ".py":
        raise ValueError(f"{file_path} is not a valid python script.")

    module_name = "runtimeFile" + ".".join(file_path.with_suffix("").parts)

    warnings.filterwarnings("default", category=DeprecationWarning, module=module_name)

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ValueError("Failed to create ModuleSpec")
        elif spec.loader is None:
            raise RuntimeError("ModuleSpec has no loader")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        sys.path.insert(0, str(file_path.parent.absolute()))
        spec.loader.exec_module(module)

    except Exception as e:
        raise RuntimeError("Module creation from file failed") from e
    else:
        return module


def search_classes_from_module(
    module: types.ModuleType, class_type: type[T]
) -> list[type[T]]:
    """Search and return all occurrence of specified class-type.

    Parameters
    -----------
    module
        Module object
    class_type
        Type of class
    """

    def is_child_scene(obj: Any) -> bool:
        return (
            isinstance(obj, type)
            and issubclass(obj, class_type)
            and obj != class_type
            and obj.__module__.startswith(module.__name__)
        )

    classes = [member for __void, member in inspect.getmembers(module, is_child_scene)]

    if len(classes) == 0:
        raise ValueError(f"Could not found any classes of type {class_type.__name__}")
    return classes


def scene_classes_for_gui(file_path: str | Path, class_type: type[T]) -> list[type[T]]:
    """Special interface only for dearpyGUI to fetch Scene-class instances.

    Parameters
    -----------
    path
        file path
    class_type
        Type of class
    """
    module = module_from_file(Path(file_path))
    return search_classes_from_module(module, class_type)
