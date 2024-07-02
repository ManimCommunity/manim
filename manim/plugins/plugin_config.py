from __future__ import annotations

from pydantic import BaseModel

from manim.renderer.opengl_renderer import OpenGLRenderer
from manim.renderer.renderer import RendererProtocol

__all__ = ("plugins",)


class PluginConfig(BaseModel):
    """Plugin abilities that should be customizable by the user.

    Examples
    --------

        .. code-block:: pycon

            >>> from manim import plugins
            >>> plugins.renderer.__name__
            'OpenGLRenderer'
            >>> class MyRenderer(OpenGLRenderer):
            ...     '''My custom renderer
            ...
            ...     All this actually has to do is implement
            ...     the RendererProtocol.
            ...     '''
            >>> plugins.renderer = MyRenderer
            >>> plugins.renderer.__name__
            'MyRenderer'
            >>> plugins.renderer = 3
            Traceback (most recent call last):
                ...
            pydantic_core._pydantic_core.ValidationError: 1 validation error for PluginConfig
            renderer
            Input should be a subclass of RendererProtocol [type=is_subclass_of, input_value=3, input_type=int]
                For further information visit https://errors.pydantic.dev/2.8/v/is_subclass_of
    """

    class Config:
        # runtime check Protocols (must be runtime_checkable Protocols)
        allow_arbitrary_types = True
        # validate setting attributes
        validate_assignment = True

    renderer: type[RendererProtocol]


plugins = PluginConfig(
    renderer=OpenGLRenderer,
)
