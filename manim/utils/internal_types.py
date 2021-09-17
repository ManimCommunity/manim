from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    from ..mobject.mobject import Group, Mobject
    from ..mobject.opengl_mobject import OpenGLGroup
    from ..mobject.types.opengl_vectorized_mobject import OpenGLVGroup
    from ..mobject.types.vectorized_mobject import VGroup

manim_group = Union["Group", "VGroup", "OpenGLGroup", "OpenGLVGroup"]
rate_function = Callable[[float], float]

__all__ = ["manim_group", "rate_function"]
