from fractions import Fraction
from unittest.mock import Mock, call

import av
import numpy as np
import pytest

from manim._config.video_encoder import VideoEncoderSpec
from manim.scene.video_segment_encoder import VideoSegmentEncoder


def _spec(*, width=4, height=2):
    return VideoEncoderSpec(
        container_format="mp4",
        codec="libx264",
        pixel_format="yuv420p",
        width=width,
        height=height,
        frame_rate=Fraction(30, 1),
        options=(("crf", "23"),),
    )


def _detached_encoder(tmp_path, *, stream=None, container=None):
    encoder = object.__new__(VideoSegmentEncoder)
    encoder.target = tmp_path / "segment.mp4"
    encoder.spec = _spec()
    encoder._next_pts = 0
    encoder._closed = False
    encoder._stream = Mock() if stream is None else stream
    encoder._container = Mock() if container is None else container
    return encoder


def _frame():
    return np.full((2, 4, 4), 127, dtype=np.uint8)


def test_write_frame_owns_repeat_and_segment_local_pts(tmp_path):
    packet = object()
    encoded_frames = []
    stream = Mock()

    def encode(frame):
        encoded_frames.append(frame)
        return [packet]

    stream.encode.side_effect = encode
    container = Mock()
    encoder = _detached_encoder(tmp_path, stream=stream, container=container)

    encoder.write_frame(_frame(), repeat=3)

    assert [frame.pts for frame in encoded_frames] == [0, 1, 2]
    assert [frame.time_base for frame in encoded_frames] == [Fraction(1, 30)] * 3
    assert encoder._next_pts == 3
    assert container.mux.call_args_list == [call(packet)] * 3


@pytest.mark.parametrize(
    ("frame", "repeat", "exception", "message"),
    [
        (np.zeros((3, 4, 4), dtype=np.uint8), 1, ValueError, "shape"),
        (np.zeros((2, 4, 3), dtype=np.uint8), 1, ValueError, "shape"),
        (np.zeros((2, 4, 4), dtype=np.float32), 1, TypeError, "uint8"),
        (np.zeros((2, 4, 4), dtype=np.uint8)[:, ::-1], 1, ValueError, "C-contiguous"),
        (np.zeros((2, 4, 4), dtype=np.uint8), 0, ValueError, "positive integer"),
        (np.zeros((2, 4, 4), dtype=np.uint8), True, ValueError, "positive integer"),
    ],
)
def test_write_frame_validates_boundary(
    tmp_path,
    frame,
    repeat,
    exception,
    message,
):
    encoder = _detached_encoder(tmp_path)

    with pytest.raises(exception, match=message):
        encoder.write_frame(frame, repeat=repeat)

    encoder._stream.encode.assert_not_called()


def test_write_frame_rejects_closed_encoder(tmp_path):
    encoder = _detached_encoder(tmp_path)
    encoder._closed = True

    with pytest.raises(RuntimeError, match="is closed"):
        encoder.write_frame(_frame())


def test_encode_failure_has_target_profile_and_original_cause(tmp_path):
    expected_exception = RuntimeError("codec exploded")
    stream = Mock()
    stream.encode.side_effect = expected_exception
    encoder = _detached_encoder(tmp_path, stream=stream)

    with pytest.raises(
        RuntimeError, match=r"segment\.mp4.*mp4/libx264/yuv420p"
    ) as exc_info:
        encoder.write_frame(_frame())

    assert exc_info.value.__cause__ is expected_exception


def test_finish_flushes_closes_and_is_idempotent(tmp_path):
    packet = object()
    stream = Mock()
    stream.encode.return_value = [packet]
    container = Mock()
    encoder = _detached_encoder(tmp_path, stream=stream, container=container)

    encoder.finish()
    encoder.finish()

    stream.encode.assert_called_once_with()
    container.mux.assert_called_once_with(packet)
    container.close.assert_called_once_with()


def test_finish_preserves_flush_failure_when_close_also_fails(tmp_path):
    flush_exception = RuntimeError("flush failed")
    close_exception = RuntimeError("close failed")
    stream = Mock()
    stream.encode.side_effect = flush_exception
    container = Mock()
    container.close.side_effect = close_exception
    encoder = _detached_encoder(tmp_path, stream=stream, container=container)

    with pytest.raises(RuntimeError, match="Failed to finish") as exc_info:
        encoder.finish()

    assert exc_info.value.__cause__ is flush_exception
    container.close.assert_called_once_with()


def test_abort_closes_removes_target_and_is_idempotent(tmp_path):
    encoder = _detached_encoder(tmp_path)
    encoder.target.write_bytes(b"incomplete")

    encoder.abort()
    encoder.abort()

    encoder._container.close.assert_called_once_with()
    assert not encoder.target.exists()


def test_open_failure_has_target_profile_and_original_cause(tmp_path, monkeypatch):
    expected_exception = RuntimeError("open failed")
    monkeypatch.setattr(av, "open", Mock(side_effect=expected_exception))

    with pytest.raises(
        RuntimeError, match=r"segment\.mp4.*mp4/libx264/yuv420p"
    ) as exc_info:
        VideoSegmentEncoder(target=tmp_path / "segment.mp4", spec=_spec())

    assert exc_info.value.__cause__ is expected_exception
