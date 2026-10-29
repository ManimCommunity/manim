"""Decide which play calls are part of the rendered output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import config
from ..utils.exceptions import EndSceneEarlyException

if TYPE_CHECKING:
    from ..manager import Manager

__all__: list[str] = []


class _OutputSelection:
    """Which plays the rendered output contains.

    A play is left out of the output when the scene was created with
    ``skip_animations=True``, when the output is a still image, while a section
    with ``skip_animations=True`` is active, or when it lies before
    ``config.from_animation_number``. A play after
    ``config.upto_animation_number`` ends the scene.
    """

    def __init__(self, manager: Manager, *, everything: bool = False) -> None:
        self._manager = manager
        self._everything = everything
        self._section_skipped = False

    @classmethod
    def everything(cls, manager: Manager) -> _OutputSelection:
        """A selection that includes every play and never ends the scene early."""
        return cls(manager, everything=True)

    def includes(self, index: int) -> bool:
        """Return whether the play with this index is part of the output."""
        if self._everything:
            return True
        manager = self._manager
        return not (
            manager._execution.original_skipping_status
            or manager.output_spec.is_still
            or self._section_skipped
            or index < config.from_animation_number
        )

    def start_play(self, index: int) -> bool:
        """Return whether the play with this index is part of the output.

        Raises
        ------
        EndSceneEarlyException
            If the play lies after ``config.upto_animation_number``.
        """
        if not self._everything and 0 <= config.upto_animation_number < index:
            raise EndSceneEarlyException()
        return self.includes(index)

    def start_section(self, skip_animations: bool) -> None:
        self._section_skipped = skip_animations
