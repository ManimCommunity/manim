from __future__ import annotations

import os

from manim.utils import caching
from manim.utils.caching import clear_segment_cache, prune_segment_cache


def test_prune_segment_cache_ignores_non_segments_and_hidden_files(tmp_path):
    segment = tmp_path / "00001.mp4"
    segment.touch()
    ignored = [
        tmp_path / ".DS_Store",
        tmp_path / "._00001.mp4",
        tmp_path / "partial_movie_file_list.txt",
        tmp_path / "notes.txt",
    ]
    for path in ignored:
        path.touch()

    prune_segment_cache(tmp_path, 1)
    assert segment.exists()

    prune_segment_cache(tmp_path, 0)
    assert not segment.exists()
    assert all(path.exists() for path in ignored)


def test_prune_segment_cache_removes_least_recently_accessed(tmp_path):
    oldest = tmp_path / "oldest.mp4"
    newest = tmp_path / "newest.mp4"
    oldest.touch()
    newest.touch()
    os.utime(oldest, (1, 1))
    os.utime(newest, (2, 2))

    prune_segment_cache(tmp_path, 1)

    assert not oldest.exists()
    assert newest.exists()


def test_prune_segment_cache_treats_minus_one_as_unlimited(tmp_path):
    segments = [tmp_path / f"{index}.mp4" for index in range(3)]
    for segment in segments:
        segment.touch()

    prune_segment_cache(tmp_path, -1)

    assert all(segment.exists() for segment in segments)


def test_prune_segment_cache_tolerates_vanishing_files(tmp_path, monkeypatch):
    survivor = tmp_path / "00001.mp4"
    survivor.touch()
    ghost = tmp_path / "00002.mp4"
    monkeypatch.setattr(
        caching,
        "_segment_cache_files",
        lambda directory: [survivor, ghost],
    )

    prune_segment_cache(tmp_path, 0)

    assert not survivor.exists()


def test_prune_segment_cache_does_not_over_evict_for_vanished_file(
    tmp_path,
    monkeypatch,
):
    survivors = [tmp_path / f"{index:05}.mp4" for index in range(2)]
    for survivor in survivors:
        survivor.touch()
    ghost = tmp_path / "00002.mp4"
    monkeypatch.setattr(
        caching,
        "_segment_cache_files",
        lambda directory: [*survivors, ghost],
    )

    prune_segment_cache(tmp_path, len(survivors))

    assert all(survivor.exists() for survivor in survivors)


def test_clear_segment_cache_removes_only_recognized_segments(tmp_path):
    segments = [tmp_path / name for name in ("one.mp4", "two.mov", "three.webm")]
    ignored = [
        tmp_path / ".hidden.mp4",
        tmp_path / "partial_movie_file_list.txt",
        tmp_path / "unrelated.mkv",
    ]
    for path in [*segments, *ignored]:
        path.touch()

    assert clear_segment_cache(tmp_path) == 3
    assert all(not segment.exists() for segment in segments)
    assert all(path.exists() for path in ignored)


def test_clear_segment_cache_accepts_missing_directory(tmp_path):
    assert clear_segment_cache(tmp_path / "missing") == 0
