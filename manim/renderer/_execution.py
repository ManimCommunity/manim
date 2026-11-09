"""Renderer accessors for the animation clock and play state.

Before a manager is attached, the renderer stores the initial values. The manager
takes over that state object; renderer properties then read and update it through
the manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import config

if TYPE_CHECKING:
    from manim.manager import Manager


@dataclass
class _ExecutionState:
    """The animation clock and the play counters of one scene."""

    frame_rate: float
    """Frames per second of the clock: the scene's session frame rate."""
    frame: int = 0
    """The number of frames played so far: the clock."""
    time_offset: float = 0.0
    """Added to the frame time; only assigning :attr:`time` changes it."""
    num_plays: int = 0
    skip_animations: bool = False
    """Whether the current play draws no frames (see :attr:`._Play.drawn`)."""
    animations_hashes: list[str | None] = field(default_factory=list)

    @property
    def time(self) -> float:
        """The elapsed animation time in seconds."""
        return self.time_offset + self.frame / self.frame_rate

    @time.setter
    def time(self, value: float) -> None:
        self.time_offset = value - self.frame / self.frame_rate


class _RendererExecutionView:
    """Keep renderer properties working with the manager's playback state."""

    def _initialize_execution(self, skip_animations: bool) -> None:
        self._execution_owner: Manager | None = None
        # The configured frame rate, which the scene's session saves; the
        # Manager checks that they agree before it runs anything.
        self._pending_execution: _ExecutionState | None = _ExecutionState(
            frame_rate=float(config.frame_rate), skip_animations=skip_animations
        )
        #: Whether the scene leaves every play out of the output, as
        #: ``Scene(skip_animations=True)`` asks; the Manager reads it.
        self._skips_all_plays = skip_animations

    @property
    def _execution_state(self) -> _ExecutionState:
        if self._execution_owner is not None:
            return self._execution_owner._execution
        assert self._pending_execution is not None
        return self._pending_execution

    def _claim_execution(self, manager: Manager, frame_rate: float) -> _ExecutionState:
        """Hand the state to ``manager``, whose clock runs at ``frame_rate``."""
        assert self._execution_owner is None
        state = self._execution_state
        state.frame_rate = frame_rate
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
    def animations_hashes(self) -> list[str | None]:
        return self._execution_state.animations_hashes

    @animations_hashes.setter
    def animations_hashes(self, value: list[str | None]) -> None:
        self._execution_state.animations_hashes = value
