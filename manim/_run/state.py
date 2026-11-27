"""The animation clock and play state of a run, owned by the Manager."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    """The number of finished play and wait calls: the index of the next play, as
    ``-n``/``--from_animation_number`` selects plays by it."""
    skip_animations: bool = False
    """Whether the current play draws no frames (see :attr:`._Play.drawn`)."""
    animations_hashes: list[str | None] = field(default_factory=list)
    """The cache key of every play so far; ``None`` for plays left out of the
    output."""

    @property
    def time(self) -> float:
        """The elapsed animation time in seconds."""
        return self.time_offset + self.frame / self.frame_rate

    @time.setter
    def time(self, value: float) -> None:
        self.time_offset = value - self.frame / self.frame_rate
