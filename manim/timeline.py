"""Experimental timeline of the play, wait, section, caption, and sound calls of a scene."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

__all__ = ["Timeline"]

_SCHEMA = "manim.timeline"
_VERSION = 1


def _dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True, slots=True)
class Timeline:
    """A timeline recorded while a scene was evaluated or rendered.

    Get a timeline from :attr:`.Manager.timeline` after calling
    :meth:`.Manager.evaluate` or :meth:`.Manager.render` with
    ``capture_timeline=True``, or load one with :meth:`from_json`. A timeline
    cannot be modified; use :meth:`to_dict` to get its data as a dictionary you
    can edit.

    The format and Python API are experimental and may change without a
    deprecation period. See :doc:`/guides/timeline` for a description of the
    format.

    Parameters
    ----------
    _document
        The timeline as JSON text. Use :meth:`from_json` to load a timeline.
    """

    _document: str

    def __post_init__(self) -> None:
        data = json.loads(self._document)
        if (
            not isinstance(data, dict)
            or data.get("schema") != _SCHEMA
            or data.get("version") != _VERSION
        ):
            raise ValueError(f"Expected a {_SCHEMA} document with version {_VERSION}.")

    @classmethod
    def from_json(cls, document: str) -> Timeline:
        """Load a timeline from JSON text.

        Parameters
        ----------
        document
            The JSON text of a timeline, for example as written by :meth:`write`.

        Raises
        ------
        ValueError
            If the document is not a version 1 timeline.
        """
        return cls(_dumps(json.loads(document)))

    def to_dict(self) -> dict[str, Any]:
        """Return the timeline data as a new dictionary."""
        return cast("dict[str, Any]", json.loads(self._document))

    def to_json(self) -> str:
        """Return the timeline as JSON text with a trailing newline.

        The same timeline always gives the same text.
        """
        return self._document + "\n"

    def write(self, path: str | Path) -> None:
        """Save the timeline as a UTF-8 JSON file.

        An existing file is replaced only after the new file has been written
        completely.

        Parameters
        ----------
        path
            The destination file. Missing parent directories are created.
        """
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
