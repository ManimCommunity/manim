"""Decorator for Mobject-spcific Animation overrides."""


__all__ = ["override_animation"]

from typing import Type

from manim.utils.exceptions import MultiAnimationOverrideException

from .. import logger
from ..animation.animation import Animation
from ..mobject.mobject import Mobject


def _all_subclasses(cls: Type, root: bool = True):
    lst = cls.__subclasses__() + [
        s for c in cls.__subclasses__() for s in _all_subclasses(c, False)
    ]
    if root:
        return list(dict.fromkeys([cls] + lst))
    else:
        return lst


def _setup(update_docs: bool = False):
    overrides = {}
    for mobject_class in _all_subclasses(Mobject):
        for method_name in dir(mobject_class):
            method = getattr(mobject_class, method_name)
            if hasattr(method, "_override_animation"):
                animation_class = method._override_animation

                if animation_class not in overrides:
                    overrides[animation_class] = {}
                if mobject_class in overrides[animation_class]:
                    msg = (
                        f"The animation {animation_class.__name__} for "
                        f"{mobject_class.__name__} is overridden by more than one method:"
                        f" {overrides[animation_class][mobject_class].__qualname__} "
                        f"and {method.__qualname__}. If one of these methods is "
                        f"inherited make sure they are named equally."
                    )
                    logger.error(msg)
                    raise MultiAnimationOverrideException()

                # method.__doc__ = (
                #     f"{method.__doc__}\n\nNotes\n-----\n\n.. note::\n\n  Test"
                # )
                overrides[animation_class][mobject_class] = method
    Animation.overrides = overrides


def override_animation(animationClass):
    def decorator(func):
        func._override_animation = animationClass
        return func

    return decorator


override_animation.setup = _setup
