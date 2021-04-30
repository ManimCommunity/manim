"""Decorator for Mobject-specific Animation overrides."""


__all__ = ["override_animation", "setup_animation_overriding"]

from typing import Callable, List, Type

from manim.utils.exceptions import MultiAnimationOverrideException

from .. import logger
from ..animation.animation import Animation
from ..mobject.mobject import Mobject


def _all_subclasses(cls: Type, initial_call: bool = True) -> List[Type]:
    """List of all classes inheriting from a given class.

    The given class is included. The returned list is ordered breadth first and unique.

    Parameters
    ----------
    cls
        The root class whose subclasses should be retrieved.
    root
        Weather it is the initial_function call.

    Returns
    -------
    List[Type]
        All classes derived from the given class.
    """
    lst = cls.__subclasses__() + [
        s for c in cls.__subclasses__() for s in _all_subclasses(c, False)
    ]
    if initial_call:
        return list(dict.fromkeys([cls] + lst))
    else:
        return lst


def setup_animation_overriding(update_docs: bool = False):
    """Sets up :class:`~.Animation` overrides for all :class:`~.Mobject` subclasses.

    Must be called to enable the functionality of methods decorated with ``@``
    :func:`override_animation`.

    Parameters
    ----------
    update_docs
        Whether to update the docstring of overridden animations and overriding methods
        accordingly.

    Raises
    ------
    MultiAnimationOverrideException
        If one class defines multiple overrides for a single ``Animation``.
    
    """
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


def override_animation(
    animationClass: Type[Animation],
) -> Callable[[Callable], Callable]:
    """Decorator used to mark methods as overrides for specific :class:`~.Animation` types.

    Should only be used to decorate methods of classes derived from :class:`~.Mobject`.
    ``Animation`` overrides get inherited to subclasses of the ``Mobject`` who defined
    them. They don't override subclasses of the ``Animation`` they override.   

    Parameters
    ----------
    animationClass
        The animation to be overridden.

    Returns
    -------
    Callable[[Callable], Callable]
        The actual decorator. This marks the method as overriding an animation.

    Notes
    -----
    To enable the animation overriding, :func:`setup_animation_overriding` has to be
    called after the class has been fully initiated. This is automatically done for
    mobject imported from manim.
    
    Examples
    --------

    .. manim:: OverrideAnimationExample

        class MySquare(Square):
            @override_animation(FadeIn)
            def fade_in(self, **kwargs):
                return Create(self, **kwargs)
    
        setup_animation_overriding()

        class OverrideAnimationExample(Scene):
            def construct(self):
                self.play(FadeIn(MySquare()))
                
    """
    def decorator(func):
        func._override_animation = animationClass
        return func

    return decorator