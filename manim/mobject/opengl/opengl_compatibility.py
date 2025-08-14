from __future__ import annotations

from abc import ABCMeta
from typing import TypeVar

from manim import config
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_point_cloud_mobject import OpenGLPMobject
from manim.mobject.opengl.opengl_surface import OpenGLSurface
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject

from ...constants import RendererType

__all__ = ["ConvertToOpenGL"]


_Class = TypeVar("_Class", bound=type)


class ConvertToOpenGL(ABCMeta):
    """Metaclass for swapping (V)Mobject with its OpenGL counterpart at runtime
    depending on config.renderer. This metaclass should only need to be inherited
    on the lowest order inheritance classes such as Mobject and VMobject.
    """

    _converted_classes: list[ConvertToOpenGL] = []

    def __new__(
        mcls: _Class, name: str, bases: tuple[type, ...], namespace: dict[str, object]
    ) -> _Class:  # this would ideally be something like `_Type & OpenGLMobject`, but we don't have intersections yet
        if config.renderer == RendererType.OPENGL:
            # Must check class names to prevent
            # cyclic importing.
            base_names_to_opengl = {
                "Mobject": OpenGLMobject,
                "VMobject": OpenGLVMobject,
                "PMobject": OpenGLPMobject,
                "Mobject1D": OpenGLPMobject,
                "Mobject2D": OpenGLPMobject,
                "Surface": OpenGLSurface,
            }

            bases = tuple(
                base_names_to_opengl.get(base.__name__, base) for base in bases
            )

        return super().__new__(mcls, name, bases, namespace)

    def __init__(
        cls, name: str, bases: tuple[type, ...], namespace: dict[str, object]
    ) -> None:
        super().__init__(name, bases, namespace)
        cls._converted_classes.append(cls)
