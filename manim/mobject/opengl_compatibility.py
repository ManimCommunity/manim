from abc import ABCMeta

from .. import config
from .opengl_mobject import OpenGLMobject
from .opengl_three_dimensions import OpenGLSurface
from .types.opengl_point_cloud_mobject import OpenGLPMobject
from .types.opengl_vectorized_mobject import OpenGLVMobject


class ConvertToOpenGL(ABCMeta):
    """Metaclass for swapping (V)Mobject with its OpenGL counterpart at runtime
    depending on config.renderer. This metaclass should only need to be inherited
    on the lowest order inheritance classes such as Mobject and VMobject.

    Note that with this implementation, changing the value of ``config.renderer``
    after Manim has been imported won't have the desired effect and will lead to
    spurious errors.
    """

    _converted_classes = []

    def __new__(mcls, name, bases, namespace):
        if config.renderer == "opengl":
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

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._converted_classes.append(cls)
