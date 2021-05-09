"""Decorator for Mobject-specific Animation overrides."""


__all__ = ["override_animation", "_setup_animation_overriding"]

from typing import Callable, Dict, List, Type

from .. import logger
from ..mobject import Mobject
from ..utils.exceptions import MultiAnimationOverrideException


def _setup_animation_overriding(
    mobject_class: "Type(Mobject)",
    overrides: "Dict[Type[Animation], List[Dict[Type[Mobject], str]]]",
):
    """Sets up :class:`~.Animation` overrides for a :class:`~.Mobject` subclass.

    Must be called to enable the functionality of methods decorated with ``@``
    :func:`override_animation`.

    Parameters
    ----------
    mobject_class
        The Mobject subclass to add an Animation override to.
    overrides
        The overrides dict to add to.

    Raises
    ------
    MultiAnimationOverrideException
        If one class defines multiple overrides for a single ``Animation``.

    """
    for method_name in dir(mobject_class):
        if method_name.startswith("__"):  # Preventing attribute errors
            continue
        method = getattr(mobject_class, method_name)
        if hasattr(method, "_override_animation"):
            animation_class = method._override_animation

            if animation_class not in overrides:
                overrides[animation_class] = {}
            if mobject_class in overrides[animation_class]:
                msg = (
                    f"The animation {animation_class.__name__} for "
                    f"{mobject_class.__name__} is overridden by more than one method: "
                    f"{overrides[animation_class][mobject_class].__qualname__}"
                    f" and {method.__qualname__}. If one of these methods is "
                    f"inherited make sure they are named equally."
                )
                raise MultiAnimationOverrideException(msg)

            # TODO: Update docs.
            # method.__doc__ = (
            #     f"{method.__doc__}\n\nNotes\n-----\n\n.. note::\n\n  Test"
            # )
            overrides[animation_class][mobject_class] = method


def override_animation(
    animation_class: "Type[Animation]",
) -> Callable[[Callable], Callable]:
    """Decorator used to mark methods as overrides for specific :class:`~.Animation` types.

    Should only be used to decorate methods of classes derived from :class:`~.Mobject`.
    ``Animation`` overrides get inherited to subclasses of the ``Mobject`` who defined
    them. They don't override subclasses of the ``Animation`` they override.

    Parameters
    ----------
    animation_class
        The animation to be overridden.

    Returns
    -------
    Callable[[Callable], Callable]
        The actual decorator. This marks the method as overriding an animation.

    Examples
    --------

    .. manim:: OverrideAnimationExample

        class MySquare(Square):
            @override_animation(FadeIn)
            def fade_in(self, **kwargs):
                return Create(self, **kwargs)

        class OverrideAnimationExample(Scene):
            def construct(self):
                self.play(FadeIn(MySquare()))

    """

    def decorator(func):
        func._override_animation = animation_class
        return func

    return decorator
