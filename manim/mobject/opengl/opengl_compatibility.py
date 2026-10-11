from __future__ import annotations

import os
from abc import ABCMeta
from typing import Any

from manim._config import config
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_point_cloud_mobject import OpenGLPMobject
from manim.mobject.opengl.opengl_three_dimensions import OpenGLSurface
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject

from ...constants import RendererType

__all__ = ["ConvertToOpenGL", "uses_opengl_mobjects"]

#: Development switch while the OpenGL renderer learns to draw the common mobject
#: classes: with ``MANIM_OPENGL_COMMON_MOBJECTS=1``, :class:`ConvertToOpenGL` does
#: nothing and OpenGL scenes contain the same mobjects as Cairo scenes.
COMMON_MOBJECTS = os.environ.get("MANIM_OPENGL_COMMON_MOBJECTS", "") not in ("", "0")


def uses_opengl_mobjects() -> bool:
    """Whether mobjects are created from the OpenGL classes.

    True for the OpenGL renderer unless :data:`COMMON_MOBJECTS` is set. Code that
    depends on the mobject classes asks this instead of checking the renderer.
    """
    return config.renderer == RendererType.OPENGL and not COMMON_MOBJECTS


class ConvertToOpenGL(ABCMeta):
    """Metaclass for swapping (V)Mobject with its OpenGL counterpart at runtime
    depending on config.renderer. This metaclass should only need to be inherited
    on the lowest order inheritance classes such as Mobject and VMobject.
    """

    _converted_classes: list[type] = []

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]
    ) -> type:
        if uses_opengl_mobjects():
            # Must check class names to prevent
            # cyclic importing.
            base_names_to_opengl: dict[str, type] = {
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

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        super().__init__(name, bases, namespace)
        cls._converted_classes.append(cls)
