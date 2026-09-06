"""Legacy renderer views of Manager-owned execution state.

An unclaimed renderer retains bootstrap values for constructor compatibility. Once a
Manager claims them, only that Manager stores the live state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manim.manager import Manager


@dataclass
class _ExecutionState:
    time: float = 0.0
    num_plays: int = 0
    skip_animations: bool = False
    original_skipping_status: bool = False
    animations_hashes: list[str | None] = field(default_factory=list)


class _RendererExecutionView:
    """Compatibility properties, not a second execution owner or scheduler."""

    def _initialize_execution(self, skip_animations: bool) -> None:
        self._execution_owner: Manager | None = None
        self._pending_execution: _ExecutionState | None = _ExecutionState(
            skip_animations=skip_animations,
            original_skipping_status=skip_animations,
        )

    @property
    def _execution_state(self) -> _ExecutionState:
        if self._execution_owner is not None:
            return self._execution_owner._execution
        assert self._pending_execution is not None
        return self._pending_execution

    def _claim_execution(self, manager: Manager) -> _ExecutionState:
        assert self._execution_owner is None
        state = self._execution_state
        self._execution_owner = manager
        self._pending_execution = None
        return state

    @property
    def time(self) -> float:
        return self._execution_state.time

    @time.setter
    def time(self, value: float) -> None:
        self._execution_state.time = value

    @property
    def num_plays(self) -> int:
        return self._execution_state.num_plays

    @num_plays.setter
    def num_plays(self, value: int) -> None:
        self._execution_state.num_plays = value

    @property
    def skip_animations(self) -> bool:
        return self._execution_state.skip_animations

    @skip_animations.setter
    def skip_animations(self, value: bool) -> None:
        self._execution_state.skip_animations = value

    @property
    def _original_skipping_status(self) -> bool:
        return self._execution_state.original_skipping_status

    @_original_skipping_status.setter
    def _original_skipping_status(self, value: bool) -> None:
        self._execution_state.original_skipping_status = value

    @property
    def animations_hashes(self) -> list[str | None]:
        return self._execution_state.animations_hashes

    @animations_hashes.setter
    def animations_hashes(self, value: list[str | None]) -> None:
        self._execution_state.animations_hashes = value
