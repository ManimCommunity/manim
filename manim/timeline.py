"""Experimental version-1 observations of completed no-raster execution."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

__all__ = ["Timeline"]

_SCHEMA = "manim.execution-timeline"
_VERSION = 1
_PACKAGE = Path(__file__).resolve().parent


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class Timeline:
    """Immutable completed observation; dictionary access always returns a fresh copy.

    Version 1 describes no-raster evaluation only, not output frames or arbitrary
    Python dependencies. The canonical document includes a SHA-256 content revision.
    """

    _document: str

    def __post_init__(self) -> None:
        data = json.loads(self._document)
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data.get("complete") is not True
        ):
            raise ValueError("Unsupported execution timeline document.")
        if (data.get("schema"), data.get("version"), data.get("complete")) != (
            _SCHEMA,
            _VERSION,
            True,
        ):
            raise ValueError("Unsupported or incomplete execution timeline.")
        revision = data.pop("revision", None)
        if revision != hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest():
            raise ValueError("Timeline content revision does not match its document.")
        data["revision"] = revision
        object.__setattr__(self, "_document", _canonical(data))

    @classmethod
    def from_json(cls, document: str) -> Timeline:
        """Read a completed version-1 document and verify its content revision."""
        return cls(_canonical(json.loads(document)))

    def to_dict(self) -> dict[str, Any]:
        return cast("dict[str, Any]", json.loads(self._document))

    def to_json(self) -> str:
        return self._document + "\n"

    @property
    def revision(self) -> str:
        return str(self.to_dict()["revision"])

    def write(self, path: str | Path) -> None:
        """Atomically replace explicit metadata output after successful capture."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(self.to_json().encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink()


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    path: Path | None
    digest: str | None
    provenance: str
    content: bytes | None = field(default=None, repr=False)

    @classmethod
    def capture(cls, path: str | Path | None, provenance: str) -> _SourceSnapshot:
        if path is None:
            return cls(None, None, provenance)
        path = Path(path).resolve()
        try:
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
        except OSError:
            return cls(None, None, provenance)
        return cls(path, digest, provenance, content)

    def unchanged(self) -> bool:
        if self.path is None:
            return True
        return self.capture(self.path, self.provenance).digest == self.digest


class _TimelineRecorder:
    def __init__(
        self,
        scene: Any,
        frame_rate: float,
        start: float,
        source: _SourceSnapshot | None,
    ):
        if source is None:
            try:
                filename = inspect.getsourcefile(type(scene))
            except TypeError:
                filename = None
            source = _SourceSnapshot.capture(
                filename, "disk-at-evaluation-not-verified-loaded-code"
            )
        self.source = source
        self.root = (
            source.path.parent if source.path is not None else Path.cwd().resolve()
        )
        self.name = type(scene).__qualname__
        self.backend = type(scene.renderer).__name__
        self.rate = frame_rate
        self.start = start
        self.last_time = float(start)
        self.failed = False
        self.events: list[dict[str, Any]] = []
        self.declarations: list[dict[str, Any]] = []
        self.active: dict[str, Any] | None = None
        self.occurrences: dict[str, int] = {}
        self.order = 0
        self.compiling = False
        self.termination = "completed"

    def path_hint(self, path: str | Path) -> dict[str, Any]:
        resolved = Path(path).resolve()
        if resolved.is_relative_to(self.root):
            return {
                "path": resolved.relative_to(self.root).as_posix(),
                "outside_root": False,
            }
        return {"path": None, "display_name": resolved.name, "outside_root": True}

    def asset_hint(self, request: str) -> dict[str, Any]:
        if not Path(request).is_absolute():
            return {"request": str(request), "resolution": "unresolved"}
        return {**self.path_hint(request), "resolution": "absolute-request"}

    def site(self, kind: str) -> dict[str, Any] | None:
        frame = inspect.currentframe()
        try:
            while frame is not None:
                filename = frame.f_code.co_filename
                if filename.startswith("<"):
                    return {
                        "path": None,
                        "line": frame.f_lineno,
                        "coverage": "generated",
                        "site_id": None,
                        "occurrence": None,
                        "outside_root": True,
                    }
                if not filename.startswith("<") and not Path(
                    filename
                ).resolve().is_relative_to(_PACKAGE):
                    hint = self.path_hint(filename)
                    hint["line"] = frame.f_lineno
                    hint["coverage"] = (
                        "line-only"
                        if not hint["outside_root"]
                        else "outside-root-unverified"
                    )
                    if hint["outside_root"]:
                        hint.update(site_id=None, occurrence=None)
                    else:
                        key = hashlib.sha256(
                            _canonical(
                                [self.source.digest, hint["path"], frame.f_lineno, kind]
                            ).encode()
                        ).hexdigest()
                        occurrence = self.occurrences.get(key, 0)
                        self.occurrences[key] = occurrence + 1
                        hint.update(site_id=key, occurrence=occurrence)
                    return hint
                frame = frame.f_back
            return None
        finally:
            del frame

    def observe_time(self, time: float) -> None:
        """Validate reached clock values, including between events and at teardown."""
        if not math.isfinite(time) or time < self.last_time:
            self.failed = True
            raise ValueError(
                "Timeline capture does not support non-finite or backwards time."
            )
        self.last_time = float(time)

    def enter(self, time: float) -> None:
        self.observe_time(time)
        if self.active is not None or self.compiling:
            self.failed = True
            raise RuntimeError(
                "Recursive timed calls are unsupported during timeline capture."
            )
        self.compiling = True
        self.entry_order = self.order
        self.order += 1

    def begin(self, scene: Any, start: float, ordinal: int) -> None:
        from .animation.animation import Wait

        if self.active is not None:
            raise RuntimeError(
                "Recursive timed calls are unsupported during timeline capture."
            )
        animations = scene.animations or []
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
        event = {
            "id": f"event-{len(self.events):06}",
            "ordinal": ordinal,
            "order": self.entry_order,
            "kind": kind,
            "source": self.site(kind),
            "start": float(start),
            "nominal_duration": float(scene.duration),
            "samples": 0,
            "hold_intervals": 0,
            "animations": summaries,
        }
        self.events.append(event)
        self.active = event
        self.compiling = False

    def sample(self, time: float) -> None:
        self.observe_time(time)
        if self.active is None:
            self.failed = True
            raise RuntimeError("Timeline sample outside a timed event.")
        self.active["samples"] += 1

    def hold(self, count: int, time: float) -> None:
        self.observe_time(time)
        if self.active is None:
            self.failed = True
            raise RuntimeError("Timeline hold outside a timed event.")
        self.active["hold_intervals"] = count

    def end(self, time: float) -> None:
        assert self.active is not None
        self.observe_time(time)
        self.active["end"] = float(time)
        self.active = None

    def declare(self, kind: str, time: float, boundary: int, **values: Any) -> None:
        self.observe_time(time)
        # Freeze reached arguments now; don't observe subsequent user mutations.
        # A caller catching serialization failure must not publish an omitted request.
        try:
            values = json.loads(_canonical(values))
        except BaseException:
            self.failed = True
            raise
        self.declarations.append(
            {
                "kind": kind,
                "order": self.order,
                "at": float(time),
                "event_boundary": boundary,
                "event_id": (
                    self.active["id"]
                    if self.active
                    else f"event-{len(self.events):06}"
                    if self.compiling
                    else None
                ),
                "source": self.site(kind),
                **values,
            }
        )
        self.order += 1

    def finish(self, time: float) -> Timeline:
        if self.failed:
            raise RuntimeError(
                "Timeline capture is incomplete after an observation failure."
            )
        self.observe_time(time)
        if self.active is not None or self.compiling:
            raise RuntimeError("Timeline contains an unfinished event.")
        if not self.source.unchanged():
            raise RuntimeError("Primary source changed during timeline capture.")
        data = {
            "schema": _SCHEMA,
            "version": _VERSION,
            "complete": True,
            "policy": "no-raster-full",
            "termination": self.termination,
            "scene": {"name": self.name},
            "frame_rate": self.rate,
            "backend": self.backend,
            "start": float(self.start),
            "end": float(time),
            "source": {
                "path": self.source.path.name if self.source.path else None,
                "sha256": self.source.digest,
                "provenance": self.source.provenance,
                "coverage": "primary-file-only" if self.source.path else "unavailable",
                "outside_root_policy": "redacted-basename-only",
            },
            "events": self.events,
            "declarations": self.declarations,
        }
        data["revision"] = hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()
        return Timeline(_canonical(data))
