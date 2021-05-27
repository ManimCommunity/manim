from abc import ABCMeta

from .. import config
from .opengl_mobject import OpenGLMobject
from .types.opengl_vectorized_mobject import OpenGLVMobject


class ConvertToOpenGL(ABCMeta):
    """Metaclass for swapping (V)Mobject with its OpenGL counterpart at runtime
    depending on config.renderer. This metaclass should only need to be inherited
    on the lowest order inheritance classes such as Mobject and VMobject.

    Note that with this implementation, changing the value of ``config.renderer``
    after Manim has been imported won't have the desired effect and will lead to
    spurious errors.
    """

    def __new__(cls, name, bases, namespace):
        if config.renderer == "opengl":
            # must check class names to prevent
            #  cyclic importing
            baseNameDict = {
                "Mobject": OpenGLMobject,
                "VMobject": OpenGLVMobject,
            }

            bases = tuple((baseNameDict.get(base.__name__, base) for base in bases))

        return super().__new__(cls, name, bases, namespace)
