from manim.constants import RendererType

from ..mobject import Mobject
from ..types.vectorized_mobject import VMobject
from .opengl_compatibility import ConvertToOpenGL
from .opengl_mobject import OpenGLMobject
from .opengl_vectorized_mobject import OpenGLVMobject


def swap_converted_classes(renderer_type: RendererType) -> None:
    for cls in ConvertToOpenGL._converted_classes:
        if renderer_type is RendererType.OPENGL:
            conversion_dict = {
                Mobject: OpenGLMobject,
                VMobject: OpenGLVMobject,
            }
        else:
            conversion_dict = {
                OpenGLMobject: Mobject,
                OpenGLVMobject: VMobject,
            }

        cls.__bases__ = tuple(conversion_dict.get(base, base) for base in cls.__bases__)
