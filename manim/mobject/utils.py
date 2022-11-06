"""Utilities for working with mobjects."""

from __future__ import annotations

__all__ = [
    "get_mobject_class",
    "get_vectorized_mobject_class",
    "get_group_class",
    "get_vectorized_group_class",
]

from .._config import config
from ..constants import RendererType
from .mobject import Mobject
from .types.point_cloud_mobject import PMobject
from .types.vectorized_mobject import VMobject
from .opengl.opengl_mobject import OpenGLMobject
from .opengl.opengl_vectorized_mobject import OpenGLVMobject
from .opengl.opengl_point_cloud_mobject import OpenGLPMobject


def get_mobject_class() -> type:
    """Returns the base mobject class, depending on the currently active renderer."""
    if config.renderer == RendererType.CAIRO:
        return Mobject
    if config.renderer == RendererType.OPENGL:
        return OpenGLMobject
    raise NotImplementedError(
        "Base mobjects are not implemented for the active renderer."
    )


def get_vectorized_mobject_class() -> type:
    if config.renderer == RendererType.CAIRO:
        return VMobject
    if config.renderer == RendererType.OPENGL:
        return OpenGLVMobject
    raise NotImplementedError(
        "Vectorized mobjects are not implemented for the active renderer."
    )


def get_point_mobject_class() -> type:
    if config.renderer == RendererType.CAIRO:
        return PMobject
    if config.renderer == RendererType.OPENGL:
        return OpenGLPMobject
    raise NotImplementedError(
        "Point cloud mobjects are not implemented for the active renderer."
    )