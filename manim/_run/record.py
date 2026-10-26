"""Record what happens during a run: plays and section, caption, and sound calls."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .loop import _Caption, _RunListener, _Section

if TYPE_CHECKING:
    from ..manager import Manager
    from ..timeline import Timeline
    from .loop import _Play, _Sound

__all__: list[str] = []

_PACKAGE = Path(__file__).resolve().parents[1]


@functools.cache
def _resolve(filename: str) -> Path:
    return Path(filename).resolve()


def _source_file(scene_class: type) -> Path | None:
    """Return the file that defines a scene class, if there is one."""
    try:
        filename = inspect.getsourcefile(scene_class)
    except (OSError, TypeError):
        return None
    return None if filename is None else _resolve(filename)


class _RunRecord(_RunListener):
    """Record each play and each section, caption, and sound call of a run.

    Every run has a record. When the run captures a timeline, the record also
    checks that its values can be exported and :meth:`timeline` exports it.
    """

    def __init__(self, manager: Manager, mode: Literal["evaluate", "render"]) -> None:
        self._manager = manager
        self.mode = mode
        self.capture = False
        """Whether the run captures a timeline; set before the run starts."""
        self.entries: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._failed = False
        self._last_time = 0.0
        self._ended_early = False
        self._end: float | None = None
        source = _source_file(type(manager.scene))
        self._source = source
        self._root = source.parent if source is not None else Path.cwd().resolve()

    def _location(self) -> dict[str, Any] | None:
        """Return the file and line of the scene code that made the current call.

        Only a captured timeline needs it, because walking the stack costs time.
        """
        frame = inspect.currentframe()
        try:
            while frame is not None:
                filename = frame.f_code.co_filename
                if filename.startswith("<"):
                    return None
                path = _resolve(filename)
                if not path.is_relative_to(_PACKAGE):
                    relative = (
                        path.relative_to(self._root).as_posix()
                        if path.is_relative_to(self._root)
                        else None
                    )
                    return {"path": relative, "line": frame.f_lineno}
                frame = frame.f_back
            return None
        finally:
            # Break the reference cycle between this frame and its callers.
            del frame

    def _observe(self) -> float:
        """Return the current scene time, checking it when capturing a timeline."""
        time = self._manager.time
        if self.capture:
            time = float(time)
            if not math.isfinite(time) or time < self._last_time:
                self._failed = True
                raise ValueError(
                    "Timeline capture requires finite scene time that never goes "
                    "backwards."
                )
            self._last_time = time
        return time

    def play_started(self, play: _Play) -> None:
        from ..animation.animation import Wait

        animations = play.animations
        kind = (
            "wait"
            if len(animations) == 1 and isinstance(animations[0], Wait)
            else "play"
        )
        summaries = []
        for animation in animations:
            cls = type(animation)
            name = (
                f"{cls.__module__}.{cls.__qualname__}"
                if cls.__module__.startswith("manim.")
                else cls.__qualname__
            )
            summaries.append({"type": name, "run_time": float(animation.run_time)})
        self._active = {
            "kind": kind,
            "start": self._observe(),
            "end": None,
            "run_time": float(play.run_time),
            "frozen": play.frozen,
            "animations": summaries,
            "skipped": not play.in_output,
            "source": self._location() if self.capture else None,
        }
        self.entries.append(self._active)

    def play_finished(self, play: _Play, end_frame: int) -> None:
        assert self._active is not None
        self._active["end"] = self._observe()
        self._active = None

    def marker(self, marker: _Section | _Caption | _Sound) -> None:
        time = self._observe()
        if isinstance(marker, _Section):
            values: dict[str, Any] = {
                "kind": "section",
                "start": time,
                "name": marker.name,
                "type": marker.type,
                "skip_animations": marker.skip_animations,
            }
        elif isinstance(marker, _Caption):
            values = {
                "kind": "caption",
                "start": marker.start,
                "end": marker.end,
                "content": marker.content,
            }
        else:
            values = {
                "kind": "sound",
                "start": float(marker.start),
                "duration": None,
                "file": marker.file,
                "gain": marker.gain,
                "gain_to_background": marker.gain_to_background,
            }
        source = self._location() if self.capture else None
        self.entries.append({**values, "source": source})

    def run_finished(self, end_frame: int, ended_early: bool) -> None:
        self._end = self._observe()
        self._ended_early = ended_early

    def timeline(self) -> Timeline:
        """Export the record of a finished run as a :class:`.Timeline`."""
        from ..timeline import _SCHEMA, _VERSION, Timeline, _dumps

        if self._failed:
            raise RuntimeError("Timeline capture failed earlier during this run.")
        if self._end is None or self._active is not None:
            raise RuntimeError("Timeline capture ended during a play or wait call.")
        source = None
        if self._source is not None:
            with contextlib.suppress(OSError):
                digest = hashlib.sha256(self._source.read_bytes()).hexdigest()
                source = {"path": self._source.name, "sha256": digest}
        scene = self._manager.scene
        data = {
            "schema": _SCHEMA,
            "version": _VERSION,
            "mode": self.mode,
            "scene": type(scene).__qualname__,
            "backend": type(scene.renderer).__name__,
            "frame_rate": float(self._manager.session_spec.frame_rate),
            "duration": self._end,
            "ended_early": self._ended_early,
            "source": source,
            "entries": self.entries,
        }
        return Timeline(_dumps(data))
