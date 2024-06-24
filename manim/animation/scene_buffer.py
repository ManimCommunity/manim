from __future__ import annotations

from typing import final

from manim.mobject.mobject import Mobject

__all__ = ["SceneBuffer"]


@final
class SceneBuffer:
    """
    A "buffer" between :class:`.Scene` and :class:`.Animation`

    Operations an animation wants to do on :class:`.Scene` should be
    done here (eg. :meth:`.Scene.add`, :meth:`.Scene.remove`). The
    scene will then apply these changes at specific points (namely
    at the beginning and end of animations)

    It is the scenes job to clear the buffer in between the beginning
    and end of animations.
    """

    def __init__(self) -> None:
        self.to_remove: list[Mobject] = []
        self.to_add: list[Mobject] = []
        self.to_replace: list[tuple[Mobject, ...]] = []
        self.deferred = False

    def add(self, *mobs: Mobject) -> None:
        self._check_deferred()
        self.to_add.extend(mobs)

    def remove(self, *mobs: Mobject) -> None:
        self._check_deferred()
        self.to_remove.extend(mobs)

    def replace(self, mob: Mobject, *replacements: Mobject) -> None:
        self._check_deferred()
        self.to_replace.append((mob, *replacements))

    def clear(self) -> None:
        self.to_remove.clear()
        self.to_add.clear()

    def deferred_clear(self) -> None:
        """Clear ``self`` on next operation"""
        self.deferred = True

    def _check_deferred(self) -> None:
        if self.deferred:
            self.clear()
            self.deferred = False

    def __str__(self) -> str:
        to_add = self.to_add
        to_remove = self.to_remove
        return f"{type(self).__name__}({to_add=}, {to_remove=})"

    __repr__ = __str__
