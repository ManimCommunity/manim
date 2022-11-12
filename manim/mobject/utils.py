"""Utilities for working with mobjects."""

from __future__ import annotations

__all__ = [
    "get_mobject_class",
    "get_point_mobject_class",
    "get_vectorized_mobject_class",
]

from .._config import config
from ..constants import RendererType
from .mobject import Mobject
from .opengl.opengl_mobject import OpenGLMobject
from .opengl.opengl_point_cloud_mobject import OpenGLPMobject
from .opengl.opengl_vectorized_mobject import OpenGLVMobject
from .types.point_cloud_mobject import PMobject
from .types.vectorized_mobject import VMobject


def get_mobject_class() -> type:
    """Gets the base mobject class, depending on the currently active renderer.

    .. NOTE::

        This method is intended to be used in the code base of Manim itself
        or in plugins where code should work independent of the selected
        renderer.

    Examples
    --------

    The function has to be explicitly imported. We test that
    the name of the returned class is one of the known mobject
    base classes::

        >>> from manim.mobject.utils import get_mobject_class
        >>> get_mobject_class().__name__ in ['Mobject', 'OpenGLMobject']
        True
    """
    if config.renderer == RendererType.CAIRO:
        return Mobject
    if config.renderer == RendererType.OPENGL:
        return OpenGLMobject
    raise NotImplementedError(
        "Base mobjects are not implemented for the active renderer."
    )


def get_vectorized_mobject_class() -> type:
    """Gets the vectorized mobject class, depending on the currently
    active renderer.

    .. NOTE::

        This method is intended to be used in the code base of Manim itself
        or in plugins where code should work independent of the selected
        renderer.

    Examples
    --------

    The function has to be explicitly imported. We test that
    the name of the returned class is one of the known mobject
    base classes::

        >>> from manim.mobject.utils import get_vectorized_mobject_class
        >>> get_vectorized_mobject_class().__name__ in ['VMobject', 'OpenGLVMobject']
        True
    """
    if config.renderer == RendererType.CAIRO:
        return VMobject
    if config.renderer == RendererType.OPENGL:
        return OpenGLVMobject
    raise NotImplementedError(
        "Vectorized mobjects are not implemented for the active renderer."
    )


def get_point_mobject_class() -> type:
    """Gets the point cloud mobject class, depending on the currently
    active renderer.

    .. NOTE::

        This method is intended to be used in the code base of Manim itself
        or in plugins where code should work independent of the selected
        renderer.

    Examples
    --------

    The function has to be explicitly imported. We test that
    the name of the returned class is one of the known mobject
    base classes::

        >>> from manim.mobject.utils import get_point_mobject_class
        >>> get_point_mobject_class().__name__ in ['PMobject', 'OpenGLPMobject']
        True
    """
    if config.renderer == RendererType.CAIRO:
        return PMobject
    if config.renderer == RendererType.OPENGL:
        return OpenGLPMobject
    raise NotImplementedError(
        "Point cloud mobjects are not implemented for the active renderer."
    )
