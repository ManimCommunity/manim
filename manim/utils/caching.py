from __future__ import annotations

from pathlib import Path

from .. import logger

__all__ = [
    "clear_segment_cache",
    "prune_segment_cache",
]

_SEGMENT_EXTENSIONS = frozenset({".mov", ".mp4", ".webm"})


def _segment_cache_files(directory: Path) -> list[Path]:
    """Return recognized segment files currently present in ``directory``."""
    try:
        entries = directory.iterdir()
        return [
            entry
            for entry in entries
            if not entry.name.startswith(".")
            and entry.suffix.lower() in _SEGMENT_EXTENSIONS
            and entry.is_file()
        ]
    except FileNotFoundError:
        return []


def prune_segment_cache(directory: Path, max_files: int) -> None:
    """Remove the least recently accessed segments beyond ``max_files``.

    ``max_files=-1`` leaves the cache unlimited. Files that disappear during
    pruning count toward the number selected for removal, preventing a race
    from evicting an additional live segment.
    """
    if max_files == -1:
        return
    if max_files < 0:
        raise ValueError("max_files must be non-negative or -1 for unlimited")

    cached_segments = _segment_cache_files(directory)
    excess = len(cached_segments) - max_files
    if excess <= 0:
        return

    def access_time(path: Path) -> float:
        try:
            return path.stat().st_atime
        except FileNotFoundError:
            return float("-inf")

    for segment in sorted(cached_segments, key=access_time)[:excess]:
        segment.unlink(missing_ok=True)

    logger.info(
        "The segment cache exceeded %d files; removed %d least recently used "
        "segment(s). Change max_files_cached to adjust this limit.",
        max_files,
        excess,
    )


def clear_segment_cache(directory: Path) -> int:
    """Delete recognized segment files from ``directory`` and return the count."""
    removed = 0
    for segment in _segment_cache_files(directory):
        try:
            segment.unlink()
        except FileNotFoundError:
            continue
        removed += 1
    return removed
