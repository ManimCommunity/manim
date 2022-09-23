"""Utilities to concatenate multiple scenes into one."""

__all__ = ["concatenate"]

from functools import wraps
from typing import Callable, Tuple

from ..scene.scene import Scene


def concatenate(
    *scenes: Tuple[Scene],
    wait_between_scenes: float = 0,
    clear_between_scenes: bool = True
) -> Callable:
    """
    Concatenates multiple scenes into one.

    This function should be used as a class decorator, see example below.

    Parameters
    ----------
    scenes: Tuple[Scene]
        The scenes (classes) to concatenate.
    wait_between_scenes: float
        Wait time between two scenes.
    clear_between_scenes: bool
        If true, self.clear() is issued after each scene.

    Examples
    --------

    .. manim:: ConcatenateScenesExample

        class First(Scene):
            def construct(self):
                ...

        class Second(Scene):
            def construct(self):
                ...


        @concatenate
        class BothScenes(First, Second):
            pass

        # or
        @concatenate(First, Second)
        class BothScenesAlt:
            pass
    """

    def construct(self):
        for scene in self.scenes:
            scene.construct(self)
            scene.wait(self, wait_between_scenes)

            if clear_between_scenes:
                scene.clear(self)

    def wrapper(cls):
        @functools.wraps(cls)
        def __wrapper__(cls):
            if len(scenes) > 1:
                cls.__bases__ = scenes

            cls.scenes = cls.__bases__
            cls.construct = construct

            return cls

        return __wrapper__(cls)

    if len(scenes) == 1:
        return wrapper(scenes[0])

    return wrapper
