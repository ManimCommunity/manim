from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import av
import numpy as np
import pytest
from click.testing import CliRunner

from manim import FadeIn, Scene, Square, capture, tempconfig
from manim._config import config
from manim.cli.render.commands import render
from manim.utils.exceptions import RerunSceneException

_ENCODER_THREAD_PREFIX = "partial-movie-encoder-"
_UNIQUE_PLAYS = 6
_TOTAL_PLAYS = _UNIQUE_PLAYS + 2

_SCENE_NAME = "ParallelEncodingCacheScene"
_SCENE_SOURCE = textwrap.dedent(
    f"""\
    from manim import FadeIn, Scene, Square


    class {_SCENE_NAME}(Scene):
        def construct(self):
            for index in range({_UNIQUE_PLAYS}):
                square = Square(side_length=0.2 + index / 100)
                self.play(FadeIn(square), run_time=0.1)
                self.clear()

            self.play(FadeIn(Square(side_length=0.75)), run_time=0.1)
            self.clear()
            self.play(FadeIn(Square(side_length=0.75)), run_time=0.1)
    """
)


def _alive_encoder_threads():
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith(_ENCODER_THREAD_PREFIX) and thread.is_alive()
    ]


def _render_scene(media_dir, scene_file):
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "--max-inflight-encoders",
        "3",
        "--media_dir",
        str(media_dir),
        str(scene_file),
        _SCENE_NAME,
    ]
    _, err, exit_code = capture(command)
    assert exit_code == 0, err
    quality_directory = media_dir / "videos" / scene_file.stem / "480p15"
    return quality_directory, quality_directory / "partial_movie_files" / _SCENE_NAME


@pytest.mark.slow
def test_parallel_encoding_cache_behavior(tmp_path):
    scene_file = tmp_path / "parallel_encoding_scenes.py"
    scene_file.write_text(_SCENE_SOURCE)

    # The duplicate tail play must cache-hit within the run (one file fewer
    # than the number of plays); that hit exercises the writer's in-flight
    # lookup while the first tail file may still be encoding.
    media_dir = tmp_path / "media"
    quality_directory, partial_directory = _render_scene(media_dir, scene_file)
    partial_movies = sorted(partial_directory.glob("*.mp4"))
    assert len(partial_movies) == _TOTAL_PLAYS - 1
    assert (quality_directory / f"{_SCENE_NAME}.mp4").exists()
    for partial_movie in partial_movies:
        with av.open(partial_movie) as container:
            next(container.decode(video=0))

    partial_snapshot = {
        partial_movie.name: partial_movie.stat().st_mtime_ns
        for partial_movie in partial_movies
    }

    _render_scene(media_dir, scene_file)

    second_snapshot = {
        partial_movie.name: partial_movie.stat().st_mtime_ns
        for partial_movie in sorted(partial_directory.glob("*.mp4"))
    }
    assert second_snapshot == partial_snapshot


@pytest.mark.slow
def test_no_encoder_threads_survive_render(config, tmp_path):
    class ThreadSweepScene(Scene):
        def construct(self):
            for index in range(3):
                square = Square(side_length=0.3 + index / 10)
                self.play(FadeIn(square), run_time=0.1)
                self.clear()

    with tempconfig(
        {
            "media_dir": tmp_path,
            "quality": "low_quality",
            "max_inflight_encoders": 3,
            "encoder_queue_size": 2,
        },
    ):
        scene = ThreadSweepScene()
        scene.render()

    assert not _alive_encoder_threads()


def _frame():
    return np.zeros((4, 4, 4), dtype=np.uint8)


def _new_encode_job(
    tmp_path,
    monkeypatch,
    name,
    stream,
    container,
    frame_queue_size=8,
):
    from manim.scene.scene_file_writer import _PartialMovieEncodeJob

    job = _PartialMovieEncodeJob(
        path=tmp_path / f"{name}.mp4",
        animation_index=0,
        stream=Mock(),
        container=Mock(),
        frame_queue_size=frame_queue_size,
    )
    monkeypatch.setattr(job, "stream", stream)
    monkeypatch.setattr(job, "container", container)
    return job


def _assert_failed_join(job, expected_exception):
    job.thread.join(timeout=5)
    assert not job.thread.is_alive(), "Partial movie encoder did not finish"

    with pytest.raises(type(expected_exception)) as exc_info:
        job.join()

    assert exc_info.value is expected_exception
    assert not _alive_encoder_threads()


def test_encode_failure_propagates_and_drains_bounded_queue(
    tmp_path,
    monkeypatch,
    manim_caplog,
):
    expected_exception = RuntimeError("encode failed")
    encode_failed = threading.Event()
    stream = Mock()
    container = Mock()

    def encode(*args):
        if args:
            encode_failed.set()
            raise expected_exception
        return []

    stream.encode.side_effect = encode
    job = _new_encode_job(tmp_path, monkeypatch, "encode_failure", stream, container)
    job.put(1, _frame())
    assert encode_failed.wait(timeout=2), "Encode failure was not triggered"

    def fill_queue_and_seal():
        for _ in range(job.queue.maxsize + 1):
            job.put(1, _frame())
        job.seal()

    producer = threading.Thread(target=fill_queue_and_seal, daemon=True)
    producer.start()
    producer.join(timeout=5)
    assert not producer.is_alive(), "Producer deadlocked on the bounded queue"

    _assert_failed_join(job, expected_exception)
    container.close.assert_called_once_with()
    assert "Partial movie file written" not in manim_caplog.text


def test_flush_failure_propagates_and_closes_container(
    tmp_path,
    monkeypatch,
    manim_caplog,
):
    expected_exception = RuntimeError("flush failed")
    stream = Mock()
    container = Mock()

    def encode(*args):
        if args:
            return []
        raise expected_exception

    stream.encode.side_effect = encode
    job = _new_encode_job(tmp_path, monkeypatch, "flush_failure", stream, container)
    job.put(1, _frame())
    job.seal()

    _assert_failed_join(job, expected_exception)
    container.close.assert_called_once_with()
    assert "Partial movie file written" not in manim_caplog.text


def test_close_failure_propagates_after_close_attempt(
    tmp_path,
    monkeypatch,
    manim_caplog,
):
    expected_exception = RuntimeError("close failed")
    stream = Mock()
    stream.encode.return_value = []
    container = Mock()
    container.close.side_effect = expected_exception
    job = _new_encode_job(tmp_path, monkeypatch, "close_failure", stream, container)
    job.put(1, _frame())
    job.seal()

    _assert_failed_join(job, expected_exception)
    container.close.assert_called_once_with()
    assert "Partial movie file written" not in manim_caplog.text


def test_encode_failure_precedes_close_failure_and_removes_partial(
    tmp_path,
    monkeypatch,
    manim_caplog,
):
    expected_exception = RuntimeError("encode failed")
    close_exception = RuntimeError("close failed")
    stream = Mock()
    container = Mock()

    def encode(*args):
        if args:
            raise expected_exception
        return []

    stream.encode.side_effect = encode
    container.close.side_effect = close_exception
    job = _new_encode_job(
        tmp_path,
        monkeypatch,
        "encode_and_close_failure",
        stream,
        container,
    )
    job.path.write_bytes(b"stale")
    job.put(1, _frame())
    job.seal()

    _assert_failed_join(job, expected_exception)
    container.close.assert_called_once_with()
    assert not job.path.exists()
    assert "Partial movie file written" not in manim_caplog.text


def test_partial_cleanup_failure_does_not_mask_encode_failure(
    tmp_path,
    monkeypatch,
    manim_caplog,
):
    expected_exception = RuntimeError("encode failed")
    cleanup_exception = PermissionError("cannot remove partial")
    stream = Mock()
    container = Mock()

    def encode(*args):
        if args:
            raise expected_exception
        return []

    stream.encode.side_effect = encode
    job = _new_encode_job(tmp_path, monkeypatch, "cleanup_failure", stream, container)
    job.path.write_bytes(b"stale")
    job.put(1, _frame())
    job.seal()
    unlink = Mock(side_effect=cleanup_exception)
    monkeypatch.setattr(Path, "unlink", unlink)

    _assert_failed_join(job, expected_exception)

    unlink.assert_called_once_with(missing_ok=True)
    assert "Failed to remove incomplete partial movie file" in manim_caplog.text
    assert "cannot remove partial" in manim_caplog.text
    assert "Partial movie file written" not in manim_caplog.text


def test_successful_encode_job_logs_partial_movie_written(
    tmp_path,
    monkeypatch,
    manim_caplog,
):
    stream = Mock()
    stream.encode.return_value = []
    container = Mock()
    job = _new_encode_job(tmp_path, monkeypatch, "encode_success", stream, container)
    job.put(1, _frame())
    job.seal()

    job.join()

    container.close.assert_called_once_with()
    assert "Partial movie file written" in manim_caplog.text
    assert not _alive_encoder_threads()


def test_write_frame_fails_fast_after_encoder_failure(
    config,
    tmp_path,
    monkeypatch,
):
    from manim.scene.scene_file_writer import SceneFileWriter

    expected_exception = RuntimeError("encode failed")
    stream = Mock()
    container = Mock()

    def encode(*args):
        if args:
            raise expected_exception
        return []

    stream.encode.side_effect = encode
    config.media_dir = str(tmp_path)
    renderer = Mock()
    renderer.num_plays = 0
    writer = SceneFileWriter(renderer, "FailFastScene")
    job = _new_encode_job(tmp_path, monkeypatch, "fail_fast", stream, container)
    job.path.write_bytes(b"stale")
    writer._current_encode_job = job

    try:
        writer.write_frame(_frame())
        for _ in range(500):
            if job.failed:
                break
            time.sleep(0.01)
        assert job.failed, "Encoder failure was not captured in time"

        with pytest.raises(RuntimeError) as exc_info:
            writer.write_frame(_frame())

        assert exc_info.value is expected_exception
        assert writer._current_encode_job is None
        assert not job.path.exists()
        assert not _alive_encoder_threads()
    finally:
        # An assertion failure above must not leave an unsealed non-daemon
        # worker behind: it would hang pytest at exit.
        if writer._current_encode_job is not None:
            job.seal()
            writer._current_encode_job = None
        job.thread.join(timeout=5)


@pytest.mark.parametrize(
    ("max_inflight_encoders", "encoder_queue_size", "expected_queue_size"),
    [(1, 8, 0), (1, 3, 0), (2, 8, 8), (2, 3, 3)],
)
def test_frame_queue_configuration(
    config,
    tmp_path,
    max_inflight_encoders,
    encoder_queue_size,
    expected_queue_size,
):
    from manim.scene.scene_file_writer import SceneFileWriter

    config.max_inflight_encoders = max_inflight_encoders
    config.encoder_queue_size = encoder_queue_size
    renderer = Mock()
    renderer.num_plays = 0
    writer = SceneFileWriter(renderer, "FrameQueueSizeScene")

    writer.open_partial_movie_stream(tmp_path / "partial.mp4")
    job = writer._current_encode_job
    assert job is not None
    assert job.queue.maxsize == expected_queue_size

    writer.close_partial_movie_stream()
    writer.join_all_encode_jobs()
    assert not _alive_encoder_threads()


@pytest.mark.parametrize("max_inflight_encoders", [1, 2, 3])
def test_close_partial_movie_stream_respects_cap_and_joins_fifo(
    config,
    tmp_path,
    max_inflight_encoders,
):
    from manim.scene.scene_file_writer import SceneFileWriter

    config.max_inflight_encoders = max_inflight_encoders
    renderer = Mock()
    renderer.num_plays = 0
    writer = SceneFileWriter(renderer, "EncoderCapScene")
    jobs = [Mock(path=tmp_path / f"partial_{index}.mp4") for index in range(3)]

    for index, job in enumerate(jobs):
        writer._current_encode_job = job
        writer.close_partial_movie_stream()

        closed_jobs = jobs[: index + 1]
        expected_joined = max(
            0,
            len(closed_jobs) - max_inflight_encoders + 1,
        )
        expected_inflight = closed_jobs[expected_joined:]
        assert writer._inflight_encode_jobs == expected_inflight
        assert writer._inflight_by_path == {
            str(inflight_job.path): inflight_job for inflight_job in expected_inflight
        }
        assert len(writer._inflight_encode_jobs) < max_inflight_encoders

        for joined_job in closed_jobs[:expected_joined]:
            joined_job.join.assert_called_once_with()
            assert str(joined_job.path) not in writer._inflight_by_path
        for inflight_job in expected_inflight:
            inflight_job.join.assert_not_called()
            assert writer._inflight_by_path[str(inflight_job.path)] is inflight_job

    for job in jobs:
        job.seal.assert_called_once_with()


def test_cap_join_failure_drains_all_inflight_jobs(config, tmp_path):
    from manim.scene.scene_file_writer import SceneFileWriter

    primary_exception = RuntimeError("first join failed")
    secondary_exception = RuntimeError("second join failed")
    config.max_inflight_encoders = 3
    renderer = Mock()
    renderer.num_plays = 0
    writer = SceneFileWriter(renderer, "EncoderCapFailureScene")
    jobs = [Mock(path=tmp_path / f"partial_{index}.mp4") for index in range(3)]
    jobs[0].join.side_effect = primary_exception
    jobs[1].join.side_effect = secondary_exception

    for job in jobs[:2]:
        writer._current_encode_job = job
        writer.close_partial_movie_stream()

    writer._current_encode_job = jobs[2]
    with pytest.raises(RuntimeError) as exc_info:
        writer.close_partial_movie_stream()

    assert exc_info.value is primary_exception
    for job in jobs:
        job.seal.assert_called_once_with()
        job.join.assert_called_once_with()
    assert writer._current_encode_job is None
    assert writer._inflight_encode_jobs == []
    assert writer._inflight_by_path == {}


def test_is_already_cached_joins_same_path_inflight_job(config, tmp_path):
    from manim.scene.scene_file_writer import SceneFileWriter

    renderer = Mock()
    renderer.num_plays = 0
    writer = SceneFileWriter(renderer, "CachedInflightScene")
    hash_invocation = "same_path_hash"
    path = (
        writer.partial_movie_directory
        / f"{hash_invocation}{config['movie_file_extension']}"
    )
    job = Mock(path=path)
    writer._inflight_encode_jobs.append(job)
    writer._inflight_by_path[str(path)] = job

    writer.is_already_cached(hash_invocation)

    job.join.assert_called_once_with()
    assert writer._inflight_encode_jobs == []
    assert writer._inflight_by_path == {}


def test_same_path_join_failure_drains_unrelated_jobs(config, tmp_path):
    from manim.scene.scene_file_writer import SceneFileWriter

    expected_exception = RuntimeError("same-path join failed")
    renderer = Mock()
    renderer.num_plays = 0
    writer = SceneFileWriter(renderer, "CachedInflightFailureScene")
    hash_invocation = "failing_same_path_hash"
    path = (
        writer.partial_movie_directory
        / f"{hash_invocation}{config['movie_file_extension']}"
    )
    unrelated_job = Mock(path=tmp_path / "unrelated.mp4")
    same_path_job = Mock(path=path)
    same_path_job.join.side_effect = expected_exception
    writer._inflight_encode_jobs.extend([unrelated_job, same_path_job])
    writer._inflight_by_path[str(unrelated_job.path)] = unrelated_job
    writer._inflight_by_path[str(path)] = same_path_job

    with pytest.raises(RuntimeError) as exc_info:
        writer.is_already_cached(hash_invocation)

    assert exc_info.value is expected_exception
    same_path_job.join.assert_called_once_with()
    unrelated_job.join.assert_called_once_with()
    assert writer._inflight_encode_jobs == []
    assert writer._inflight_by_path == {}


def test_open_partial_movie_stream_joins_same_path_inflight_job(config, tmp_path):
    from manim.scene.scene_file_writer import SceneFileWriter

    renderer = Mock()
    renderer.num_plays = 0
    writer = SceneFileWriter(renderer, "OpenInflightScene")
    path = tmp_path / "same_path.mp4"
    inflight_job = Mock(path=path)
    writer._inflight_encode_jobs.append(inflight_job)
    writer._inflight_by_path[str(path)] = inflight_job

    writer.open_partial_movie_stream(file_path=path)
    current_job = writer._current_encode_job
    assert current_job is not None
    try:
        inflight_job.join.assert_called_once_with()
        assert writer._inflight_encode_jobs == []
        assert writer._inflight_by_path == {}
    finally:
        current_job.seal()
        current_job.join()
        writer._current_encode_job = None

    assert not _alive_encoder_threads()


def test_finish_propagates_join_failure_and_clears_inflight_state(
    config,
    tmp_path,
    monkeypatch,
):
    from manim.scene.scene_file_writer import SceneFileWriter

    expected_exception = RuntimeError("join failed")
    renderer = Mock()
    renderer.num_plays = 0
    writer = SceneFileWriter(renderer, "JoinFailureScene")
    failing_job = Mock(path=tmp_path / "failing.mp4")
    failing_job.join.side_effect = expected_exception
    succeeding_job = Mock(path=tmp_path / "succeeding.mp4")
    writer._inflight_encode_jobs.extend([failing_job, succeeding_job])
    writer._inflight_by_path[str(failing_job.path)] = failing_job
    writer._inflight_by_path[str(succeeding_job.path)] = succeeding_job
    combine_to_movie = Mock()
    monkeypatch.setattr(writer, "combine_to_movie", combine_to_movie)

    with pytest.raises(RuntimeError) as exc_info:
        writer.finish()

    assert exc_info.value is expected_exception
    failing_job.join.assert_called_once_with()
    succeeding_job.join.assert_called_once_with()
    assert writer._inflight_encode_jobs == []
    assert writer._inflight_by_path == {}
    combine_to_movie.assert_not_called()


def _new_writer(config, tmp_path, scene_name):
    from manim.scene.scene_file_writer import SceneFileWriter

    config.media_dir = str(tmp_path)
    renderer = Mock()
    renderer.num_plays = 0
    return SceneFileWriter(renderer, scene_name)


def _healthy_current_job(tmp_path, monkeypatch, name):
    stream = Mock()
    stream.encode.return_value = []
    job = _new_encode_job(tmp_path, monkeypatch, name, stream, Mock())
    job.path.write_bytes(b"stale")
    return job


def _add_inflight_job(writer, job):
    writer._inflight_encode_jobs.append(job)
    writer._inflight_by_path[str(job.path)] = job


def test_abort_encode_jobs_unlinks_current_and_drains_inflight(
    config,
    tmp_path,
    monkeypatch,
    manim_caplog,
):
    writer = _new_writer(config, tmp_path, "AbortScene")
    job = _healthy_current_job(tmp_path, monkeypatch, "abort_current")
    writer._current_encode_job = job
    failing_inflight = Mock(path=tmp_path / "inflight.mp4")
    failing_inflight.join.side_effect = RuntimeError("in-flight join failed")
    _add_inflight_job(writer, failing_inflight)

    writer.abort_encode_jobs()

    assert writer._current_encode_job is None
    assert not job.path.exists()
    assert writer._inflight_encode_jobs == []
    assert writer._inflight_by_path == {}
    assert not _alive_encoder_threads()
    assert "Discarded partial movie file" in manim_caplog.text
    assert "Encoder failure while aborting render" in manim_caplog.text

    # A second call must be a no-op.
    writer.abort_encode_jobs()
    assert writer._current_encode_job is None


def test_abort_encode_jobs_reraise_propagates_inflight_failure(
    config,
    tmp_path,
):
    expected_exception = RuntimeError("in-flight join failed")
    writer = _new_writer(config, tmp_path, "AbortReraiseScene")
    failing_inflight = Mock(path=tmp_path / "inflight.mp4")
    failing_inflight.join.side_effect = expected_exception
    _add_inflight_job(writer, failing_inflight)

    with pytest.raises(RuntimeError) as exc_info:
        writer.abort_encode_jobs(reraise_encoder_failures=True)

    assert exc_info.value is expected_exception
    assert writer._inflight_encode_jobs == []
    assert writer._inflight_by_path == {}


def test_abort_encode_jobs_cleanup_failure_logs_warning(
    config,
    tmp_path,
    monkeypatch,
    manim_caplog,
):
    writer = _new_writer(config, tmp_path, "AbortCleanupFailureScene")
    job = _healthy_current_job(tmp_path, monkeypatch, "abort_cleanup_failure")
    writer._current_encode_job = job
    unlink = Mock(side_effect=PermissionError("cannot remove partial"))
    monkeypatch.setattr(Path, "unlink", unlink)

    writer.abort_encode_jobs()

    assert writer._current_encode_job is None
    unlink.assert_called_once_with(missing_ok=True)
    assert "Failed to remove incomplete partial movie file" in manim_caplog.text
    assert "cannot remove partial" in manim_caplog.text
    assert "Discarded partial movie file" not in manim_caplog.text
    assert not _alive_encoder_threads()


def test_abort_encode_jobs_noop_on_dry_run_writer(config):
    from manim.scene.scene_file_writer import SceneFileWriter

    with tempconfig({"dry_run": True}):
        renderer = Mock()
        renderer.num_plays = 0
        writer = SceneFileWriter(renderer, "DryRunAbortScene")

        writer.abort_encode_jobs()
        writer.abort_encode_jobs(reraise_encoder_failures=True)

    assert not _alive_encoder_threads()


def test_keyboard_interrupt_aborts_encode_jobs_and_reraises(config):
    scene = Scene(renderer=Mock())

    def construct():
        raise KeyboardInterrupt

    scene.construct = construct

    with pytest.raises(KeyboardInterrupt):
        scene.render()

    scene.renderer.file_writer.abort_encode_jobs.assert_called_once_with()


def test_rerun_propagates_encoder_failure(config, tmp_path):
    expected_exception = RuntimeError("in-flight encoder failed")
    writer = _new_writer(config, tmp_path, "RerunFailureScene")
    failing_inflight = Mock(path=tmp_path / "inflight.mp4")
    failing_inflight.join.side_effect = expected_exception
    _add_inflight_job(writer, failing_inflight)
    scene = Scene(renderer=Mock())
    scene.renderer.file_writer = writer

    def construct():
        raise RerunSceneException

    scene.construct = construct

    with pytest.raises(RuntimeError) as exc_info:
        scene.render()

    assert exc_info.value is expected_exception
    assert writer._inflight_encode_jobs == []
    assert writer._inflight_by_path == {}


def test_rerun_propagates_failed_current_job(config, tmp_path, monkeypatch):
    expected_exception = RuntimeError("encode failed")
    stream = Mock()

    def encode(*args):
        if args:
            raise expected_exception
        return []

    stream.encode.side_effect = encode
    writer = _new_writer(config, tmp_path, "RerunCurrentFailureScene")
    job = _new_encode_job(tmp_path, monkeypatch, "rerun_current", stream, Mock())
    job.path.write_bytes(b"stale")
    writer._current_encode_job = job
    job.put(1, _frame())
    for _ in range(500):
        if job.failed:
            break
        time.sleep(0.01)

    scene = Scene(renderer=Mock())
    scene.renderer.file_writer = writer

    def construct():
        raise RerunSceneException

    scene.construct = construct

    try:
        assert job.failed, "Encoder failure was not captured in time"

        with pytest.raises(RuntimeError) as exc_info:
            scene.render()

        assert exc_info.value is expected_exception
        assert writer._current_encode_job is None
        assert not job.path.exists()
        assert not _alive_encoder_threads()
    finally:
        # An assertion failure above must not leave an unsealed non-daemon
        # worker behind: it would hang pytest at exit.
        if writer._current_encode_job is not None:
            job.seal()
            writer._current_encode_job = None
        job.thread.join(timeout=5)


@pytest.mark.slow
def test_end_scene_early_keeps_completed_partials(config, tmp_path):
    class EarlyEndScene(Scene):
        def construct(self):
            for index in range(3):
                square = Square(side_length=0.3 + index / 10)
                self.play(FadeIn(square), run_time=0.1)
                self.clear()

    with tempconfig(
        {
            "media_dir": tmp_path,
            "quality": "low_quality",
            "max_inflight_encoders": 3,
            "disable_caching": True,
            "upto_animation_number": 1,
        },
    ):
        scene = EarlyEndScene()
        scene.render()
        partial_directory = Path(scene.renderer.file_writer.partial_movie_directory)

    partial_movies = sorted(path.name for path in partial_directory.glob("*.mp4"))
    assert partial_movies == ["uncached_00000.mp4", "uncached_00001.mp4"]
    assert not _alive_encoder_threads()


_MID_PLAY_FAILURE_SCENE_NAME = "MidPlayFailureScene"
_MID_PLAY_FAILURE_SCENE_SOURCE = textwrap.dedent(
    f"""\
    from manim import FadeIn, Scene, Square


    class {_MID_PLAY_FAILURE_SCENE_NAME}(Scene):
        def construct(self):
            self.play(FadeIn(Square(side_length=0.5)), run_time=0.1)

            square = Square()

            def fail(mobject, dt):
                # compile_animation_data pre-runs updaters with dt=0 before
                # the partial movie stream opens; only raise during playback.
                if dt > 0:
                    raise RuntimeError("updater failure mid-play")

            square.add_updater(fail)
            self.add(square)
            self.wait(0.1)
    """
)


@pytest.mark.slow
def test_mid_play_exception_does_not_hang_process(tmp_path):
    scene_file = tmp_path / "mid_play_failure_scene.py"
    scene_file.write_text(_MID_PLAY_FAILURE_SCENE_SOURCE)
    media_dir = tmp_path / "media"
    command = [
        sys.executable,
        "-m",
        "manim",
        "-ql",
        "--disable_caching",
        "--max-inflight-encoders",
        "3",
        "--media_dir",
        str(media_dir),
        str(scene_file),
        _MID_PLAY_FAILURE_SCENE_NAME,
    ]

    # Without the abort path in Scene.render, the mid-play exception leaves
    # an unsealed encode job whose non-daemon worker hangs the interpreter
    # at exit; this run then dies with subprocess.TimeoutExpired.
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode != 0
    assert "updater failure mid-play" in completed.stdout + completed.stderr
    partial_directory = (
        media_dir
        / "videos"
        / scene_file.stem
        / "480p15"
        / "partial_movie_files"
        / _MID_PLAY_FAILURE_SCENE_NAME
    )
    assert (partial_directory / "uncached_00000.mp4").exists()
    assert not (partial_directory / "uncached_00001.mp4").exists()


def test_parallel_encoder_flags_digest_into_config(tmp_path):
    scene_file = tmp_path / "trivial_scene.py"
    scene_file.write_text("# never executed: --jupyter returns before rendering\n")
    cfg_file = tmp_path / "custom.cfg"
    cfg_file.write_text(
        "[CLI]\nmax_inflight_encoders = 2\nencoder_queue_size = 5\n",
    )
    runner = CliRunner()

    common_args = [str(scene_file), "--jupyter", "--config_file", str(cfg_file)]
    result = runner.invoke(
        render,
        [
            *common_args,
            "--max-inflight-encoders",
            "4",
            "--encoder-queue-size",
            "7",
        ],
        standalone_mode=False,
    )
    assert result.exit_code == 0
    with tempconfig({}):
        config.digest_args(result.return_value)
        assert config.max_inflight_encoders == 4
        assert config.encoder_queue_size == 7

    result = runner.invoke(render, common_args, standalone_mode=False)
    assert result.exit_code == 0
    with tempconfig({}):
        config.digest_args(result.return_value)
        assert config.max_inflight_encoders == 2
        assert config.encoder_queue_size == 5

    with tempconfig({}):
        assert config.max_inflight_encoders == 1
        assert config.encoder_queue_size == 8


@pytest.mark.parametrize(
    "flag",
    ["--max-inflight-encoders", "--encoder-queue-size"],
)
def test_parallel_encoder_flags_reject_non_positive(tmp_path, flag):
    scene_file = tmp_path / "trivial_scene.py"
    scene_file.write_text("# never executed: parsing fails before any render\n")
    runner = CliRunner()

    for bad_value in ["0", "-1"]:
        result = runner.invoke(render, [str(scene_file), flag, bad_value])
        assert result.exit_code == 2
        assert "Invalid value" in result.output


def test_encoder_queue_size_rejects_non_positive_programmatic_values():
    for bad_value in [0, -1]:
        with pytest.raises(ValueError, match="positive integer"):
            config.encoder_queue_size = bad_value
