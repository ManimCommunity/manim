from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from av.codec.context import CodecContext

from manim import tempconfig
from manim._config import config
from manim._config.output import OutputFormat, OutputSpec
from manim._config.output_plan import resolve_media_layout, resolve_output_plan
from manim._config.video_encoder import resolve_video_encoder
from manim.scene.scene_file_writer import SceneFileWriter, SceneFileWriterSettings

_WIDTH = 16
_HEIGHT = 12
_FRAME_RATE = 5
_FRAME_COUNT = 3


def _output(format: str, transparent: bool) -> OutputSpec:
    return OutputSpec(
        format=OutputFormat(format),
        transparent=transparent,
        save_sections=False,
        fallback_to_still=False,
    )


def _writer(output: OutputSpec) -> SceneFileWriter:
    layout = resolve_media_layout(
        config,
        output,
        module_name="segment_profiles",
        scene_name="SegmentProfileScene",
        working_directory=Path.cwd(),
    )
    plan = resolve_output_plan(
        layout,
        output,
        scene_name="SegmentProfileScene",
        requested_output_name=None,
    )
    video_encoder = resolve_video_encoder(
        output,
        width=config.pixel_width,
        height=config.pixel_height,
        frame_rate=config.frame_rate,
        codec=config.video_codec,
        pixel_format=config.pixel_format,
        options=config.video_encoder_options,
    )
    settings = SceneFileWriterSettings(
        output=output,
        plan=plan,
        video_encoder=video_encoder,
        max_inflight_encoders=config.max_inflight_encoders,
        encoder_queue_size=config.encoder_queue_size,
        max_files_cached=config.max_files_cached,
        assets_dir=Path.cwd(),
    )
    return SceneFileWriter(settings)


def _asymmetric_rgba_frame() -> np.ndarray:
    frame = np.empty((_HEIGHT, _WIDTH, 4), dtype=np.uint8)
    frame[: _HEIGHT // 2, : _WIDTH // 2] = [255, 0, 0, 32]
    frame[: _HEIGHT // 2, _WIDTH // 2 :] = [0, 255, 0, 96]
    frame[_HEIGHT // 2 :, : _WIDTH // 2] = [0, 0, 255, 160]
    frame[_HEIGHT // 2 :, _WIDTH // 2 :] = [255, 255, 255, 224]
    return frame


def _decode_frames(
    path: Path,
    *,
    transparent_vp9: bool,
) -> tuple[str, str, Fraction | None, list[np.ndarray], int]:
    with av.open(path) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        pixel_format = stream.codec_context.format.name
        rate = stream.average_rate
        audio_streams = len(container.streams.audio)

        if transparent_vp9:
            decoder = CodecContext.create("libvpx-vp9", "r")
            decoded = []
            for packet in container.demux(video=0):
                decoded.extend(decoder.decode(packet))
            pixel_format = decoded[0].format.name
        else:
            decoded = list(container.decode(video=0))

        frames = [frame.to_ndarray(format="rgba") for frame in decoded]
    return codec, pixel_format, rate, frames, audio_streams


@pytest.mark.parametrize(
    (
        "format",
        "transparent",
        "segment_extension",
        "codec",
        "pixel_format",
    ),
    [
        ("mp4", False, ".mp4", "h264", "yuv420p"),
        ("mov", False, ".mov", "h264", "yuv420p"),
        ("mov", True, ".mov", "qtrle", "argb"),
        ("webm", False, ".webm", "vp9", "yuv420p"),
        ("webm", True, ".webm", "vp9", "yuva420p"),
        ("gif", False, ".mp4", "h264", "yuv420p"),
        ("gif", True, ".mov", "qtrle", "argb"),
    ],
)
def test_cached_segment_profile_and_pixel_orientation(
    tmp_path,
    format,
    transparent,
    segment_extension,
    codec,
    pixel_format,
):
    output = _output(format, transparent)
    target = tmp_path / f"segment{segment_extension}"
    source = _asymmetric_rgba_frame()

    with tempconfig(
        {
            "media_dir": tmp_path,
            "pixel_width": _WIDTH,
            "pixel_height": _HEIGHT,
            "frame_rate": _FRAME_RATE,
        },
    ):
        writer = _writer(output)
        writer.open_partial_movie_stream(animation_index=0, file_path=target)
        writer.write_frame(source, repeat=_FRAME_COUNT)
        writer.close_partial_movie_stream()
        writer.join_all_encode_jobs()

    (
        actual_codec,
        actual_pixel_format,
        actual_rate,
        decoded_frames,
        audio_streams,
    ) = _decode_frames(
        target,
        transparent_vp9=format == "webm" and transparent,
    )

    assert output.segment_extension == segment_extension
    assert actual_codec == codec
    assert actual_pixel_format == pixel_format
    assert actual_rate == Fraction(_FRAME_RATE, 1)
    assert len(decoded_frames) == _FRAME_COUNT
    assert audio_streams == 0

    sample_points = (
        ((_HEIGHT // 4, _WIDTH // 4), np.array([255, 0, 0, 32])),
        ((_HEIGHT // 4, 3 * _WIDTH // 4), np.array([0, 255, 0, 96])),
        ((3 * _HEIGHT // 4, _WIDTH // 4), np.array([0, 0, 255, 160])),
        ((3 * _HEIGHT // 4, 3 * _WIDTH // 4), np.array([255, 255, 255, 224])),
    )
    for decoded in decoded_frames:
        for (row, column), expected in sample_points:
            expected = expected.copy()
            if not transparent:
                expected[3] = 255
            np.testing.assert_allclose(
                decoded[row, column],
                expected,
                atol=15,
            )


def test_configured_encoder_profile_is_used_for_cached_segment(tmp_path):
    output = _output("mov", transparent=False)
    target = tmp_path / "custom.mov"

    with tempconfig(
        {
            "media_dir": tmp_path,
            "pixel_width": _WIDTH,
            "pixel_height": _HEIGHT,
            "frame_rate": _FRAME_RATE,
            "video_codec": "qtrle",
            "pixel_format": "argb",
            "video_encoder_options": {},
        },
    ):
        writer = _writer(output)
        writer.open_partial_movie_stream(animation_index=0, file_path=target)
        writer.write_frame(_asymmetric_rgba_frame())
        writer.close_partial_movie_stream()
        writer.join_all_encode_jobs()

    codec, pixel_format, rate, frames, audio_streams = _decode_frames(
        target,
        transparent_vp9=False,
    )
    assert codec == "qtrle"
    assert pixel_format == "argb"
    assert rate == Fraction(_FRAME_RATE, 1)
    assert len(frames) == 1
    assert audio_streams == 0
