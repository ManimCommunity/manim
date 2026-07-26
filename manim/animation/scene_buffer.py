from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manim.mobject.opengl.opengl_mobject import OpenGLMobject as Mobject

__all__ = ["SceneBuffer", "SceneOperation"]


class SceneOperation(Enum):
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


class SceneBuffer:
    """
    A "buffer" between :class:`.Scene` and :class:`.Animation`

    Operations an animation wants to do on :class:`.Scene` should be
    done here (eg. :meth:`.Scene.add`, :meth:`.Scene.remove`). The
    scene will then apply these changes at specific points (namely
    at the beginning and end of animations)

    It is the scenes job to clear the buffer in between the beginning
    and end of animations.

    To iterate over the operations, simply iterate over the buffer.

    Example
    -------

        .. code-block:: pycon

            >>> buffer = SceneBuffer()
            >>> buffer.add(Square())
            >>> buffer.remove(Circle())
            >>> buffer.replace(Square(), Circle(), flag=True)
            >>> for operation in buffer:
            ...     print(operation)
            (SceneOperation.ADD, (Square(),), {})
            (SceneOperation.REMOVE, (Circle(),), {})
            (SceneOperation.REPLACE, (Square(), Circle()), {"flag": True})
    """

    def __init__(self) -> None:
        self.operations: list[
            tuple[SceneOperation, Sequence[Mobject], dict[str, Any]]
        ] = []

    def add(self, *mobs: Mobject, **kwargs: Any) -> None:
        """Add mobjects to the scene."""
        self.operations.append((SceneOperation.ADD, mobs, kwargs))

    def remove(self, *mobs: Mobject, **kwargs: Any) -> None:
        """Remove mobjects from the scene."""
        self.operations.append((SceneOperation.REMOVE, mobs, kwargs))

    def replace(self, mob: Mobject, *replacements: Mobject, **kwargs: Any) -> None:
        """Replace a ``mob`` with ``replacements`` on the scene."""
        self.operations.append((SceneOperation.REPLACE, (mob, *replacements), kwargs))

    def clear(self) -> None:
        """Clear the buffer."""
        self.operations.clear()

    def __str__(self) -> str:
        operations = self.operations
        return f"{type(self).__name__}({operations=})"

    __repr__ = __str__

    def __iter__(
        self,
    ) -> Iterator[tuple[SceneOperation, Sequence[Mobject], dict[str, Any]]]:
        return iter(self.operations)
