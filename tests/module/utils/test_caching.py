from __future__ import annotations

import os
from unittest.mock import Mock

import numpy as np

from manim.utils import caching
from manim.utils.caching import (
    clear_segment_cache,
    handle_caching_play,
    prune_segment_cache,
)


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


def test_opengl_cache_path_supplies_backend_encoder_and_raster_state(monkeypatch):
    encoder = object()
    fingerprint = Mock(return_value="encoder-token")
    hash_play = Mock(return_value="cache-key")
    monkeypatch.setattr(caching, "video_encoder_fingerprint", fingerprint)
    monkeypatch.setattr(caching, "get_hash_from_play_call", hash_play)

    class FakeScene:
        def __init__(self):
            self.mobjects = [object()]
            self.meshes = [object()]
            self.session_spec = Mock(video_encoder=encoder)

        def compile_animations(self, *args, **kwargs):
            return []

        def add_mobjects_from_animations(self, animations):
            pass

    class FakeRenderer:
        _original_skipping_status = False
        skip_animations = False
        num_plays = 0
        animations_hashes = []
        camera = object()
        background_color = np.array([0.1, 0.2, 0.3, 1.0])
        anti_alias_width = 1.5
        file_writer = Mock()

        def update_skipping_status(self):
            pass

        @handle_caching_play
        def play(self, scene, *args, **kwargs):
            pass

    scene = FakeScene()
    renderer = FakeRenderer()
    renderer.file_writer.is_already_cached.return_value = False

    renderer.play(scene)

    fingerprint.assert_called_once_with(encoder)
    hash_play.assert_called_once()
    assert hash_play.call_args.kwargs == {
        "backend": "opengl",
        "encoder_fingerprint": "encoder-token",
        "renderer_state": {
            "meshes": scene.meshes,
            "background_color": renderer.background_color,
            "anti_alias_width": renderer.anti_alias_width,
        },
    }
    renderer.file_writer.add_partial_movie_file.assert_called_once_with("cache-key")
