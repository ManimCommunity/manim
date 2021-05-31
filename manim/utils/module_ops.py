import importlib.util
import inspect
import os
import re
import sys
import types
import warnings
from pathlib import Path

from .. import config, console, constants, logger


def get_module(file_name: Path):
    if str(file_name) == "-":
        module = types.ModuleType("input_scenes")
        logger.info(
            "Enter the animation's code & end with an EOF (CTRL+D on Linux/Unix, CTRL+Z on Windows):"
        )
        code = sys.stdin.read()
        if not code.startswith("from manim import"):
            logger.warning(
                "Didn't find an import statement for Manim. Importing automatically..."
            )
            code = "from manim import *\n" + code
        logger.info("Rendering animation from typed code...")
        try:
            exec(code, module.__dict__)
            return module
        except Exception as e:
            logger.error(f"Failed to render scene: {str(e)}")
            sys.exit(2)
    else:
        if Path(file_name).exists():
            ext = file_name.suffix
            if ext != ".py":
                raise ValueError(f"{file_name} is not a valid Manim python script.")
            module_name = ext.replace(os.sep, ".").split(".")[-1]

            warnings.filterwarnings(
                "default", category=DeprecationWarning, module=module_name
            )

            spec = importlib.util.spec_from_file_location(module_name, file_name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            sys.path.insert(0, str(file_name.parent.absolute()))
            spec.loader.exec_module(module)
            return module
        else:
            raise FileNotFoundError(f"{file_name} not found")


def get_scene_classes_from_module(module):
    from ..scene.scene import Scene

    def is_child_scene(obj, module):
        return (
            inspect.isclass(obj)
            and issubclass(obj, Scene)
            and obj != Scene
            and obj.__module__.startswith(module.__name__)
        )

    return [
        member[1]
        for member in inspect.getmembers(module, lambda x: is_child_scene(x, module))
    ]


def get_scenes_to_render(scene_classes):
    if not scene_classes:
        logger.error(constants.NO_SCENE_MESSAGE)
        return []
    if config["write_all"]:
        return scene_classes
    result = []
    for scene_name in config["scene_names"]:
        found = False
        for scene_class in scene_classes:
            if scene_class.__name__ == scene_name:
                result.append(scene_class)
                found = True
                break
        if not found and (scene_name != ""):
            logger.error(constants.SCENE_NOT_FOUND_MESSAGE.format(scene_name))
    if result:
        return result
    return (
        [scene_classes[0]]
        if len(scene_classes) == 1
        else prompt_user_for_choice(scene_classes)
    )


def prompt_user_for_choice(scene_classes):
    num_to_class = {}
    config["write_all"] = True
    for count, scene_class in enumerate(scene_classes):
        count += 1  # start with 1 instead of 0
        name = scene_class.__name__
        console.print(f"{count}: {name}", style="logging.level.info")
        num_to_class[count] = scene_class
    try:
        user_input = console.input(
            f"[log.message] {constants.CHOOSE_NUMBER_MESSAGE} [/log.message]"
        )
        return [
            num_to_class[int(num_str)]
            for num_str in re.split(r"\s*,\s*", user_input.strip())
        ]
    except KeyError:
        logger.error(constants.INVALID_NUMBER_MESSAGE)
        sys.exit(2)
    except EOFError:
        sys.exit(1)


def scene_classes_from_file(file_path, require_single_scene=False):
    module = get_module(file_path)
    all_scene_classes = get_scene_classes_from_module(module)
    scene_classes_to_render = get_scenes_to_render(all_scene_classes)
    if require_single_scene:
        assert len(scene_classes_to_render) == 1
        return scene_classes_to_render[0]
    return scene_classes_to_render
