from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from manim.event_handler.window import WindowABC
from manim.renderer.opengl_renderer import OpenGLRenderer
from manim.renderer.renderer import RendererProtocol
from manim.renderer.opengl_renderer_window import Window

if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from manim.manager import Manager

    HookFunction: TypeAlias = Callable[[Manager], object]


__all__ = (
    "plugins",
    "Hooks",
)


class Hooks(Enum):
    POST_CONSTRUCT = "post_construct"



class PluginConfig(BaseModel):
    """Plugin abilities that should be customizable by the user.

    Parameters
    ----------
        renderer : The renderer class to use for rendering scenes.
        window: The window class to use for displaying the scene.

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
        extra = "forbid"

    renderer: type[RendererProtocol]
    window: type[WindowABC]

    # not included in pydantic because Manager is undefined
    # due to circular imports and __future__.annotations
    # instead we do validation manually via :meth:`.register`
    _hooks: dict[Hooks, list[HookFunction]] = {hook: [] for hook in Hooks}

    @property
    def hooks(self) -> dict[Hooks, list[HookFunction]]:
        return self._hooks

    def register(self, hooks: dict[Hooks, list[HookFunction]]) -> None:
        """Register hooks to run at specific points in the program."""

        for hook, functions in hooks.items():
            if not all(callable(func) for func in functions):
                raise ValueError("All hooks must be callables!")
            if not isinstance(hook, Hooks):
                raise ValueError(f"Unknown hook type {hook}, must be an instance of enum {Hooks}")
            self._hooks[hook].extend(functions)


plugins = PluginConfig(
    renderer=OpenGLRenderer,
    window=Window
)
