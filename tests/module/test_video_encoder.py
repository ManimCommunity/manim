from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from manim._config.output import OutputFormat, OutputSpec
from manim._config.video_encoder import (
    VideoEncoderSpec,
    resolve_video_encoder,
    to_av_frame_rate,
    video_encoder_fingerprint,
)


def _output(format: OutputFormat, *, transparent: bool = False) -> OutputSpec:
    return OutputSpec(
        format=format,
        transparent=transparent,
        save_sections=False,
        fallback_to_still=False,
    )


@pytest.mark.parametrize(
    (
        "format",
        "transparent",
        "container",
        "codec",
        "pixel_format",
        "options",
    ),
    [
        (OutputFormat.MP4, False, "mp4", "libx264", "yuv420p", (("crf", "23"),)),
        (OutputFormat.MOV, False, "mov", "libx264", "yuv420p", (("crf", "23"),)),
        (OutputFormat.MOV, True, "mov", "qtrle", "argb", ()),
        (
            OutputFormat.WEBM,
            False,
            "webm",
            "libvpx-vp9",
            "yuv420p",
            (("crf", "23"),),
        ),
        (
            OutputFormat.WEBM,
            True,
            "webm",
            "libvpx-vp9",
            "yuva420p",
            (("crf", "23"),),
        ),
        (OutputFormat.GIF, False, "mp4", "libx264", "yuv420p", (("crf", "23"),)),
        (OutputFormat.GIF, True, "mov", "qtrle", "argb", ()),
    ],
)
def test_default_video_encoder_profiles(
    format,
    transparent,
    container,
    codec,
    pixel_format,
    options,
):
    spec = resolve_video_encoder(
        _output(format, transparent=transparent),
        width=1920,
        height=1080,
        frame_rate=60,
    )

    assert spec == VideoEncoderSpec(
        container_format=container,
        codec=codec,
        pixel_format=pixel_format,
        width=1920,
        height=1080,
        frame_rate=Fraction(60, 1),
        options=options,
    )


@pytest.mark.parametrize(
    "format",
    [OutputFormat.NONE, OutputFormat.PNG, OutputFormat.PNG_SEQUENCE],
)
def test_non_video_output_has_no_encoder(format):
    assert (
        resolve_video_encoder(
            _output(format),
            width=0,
            height=0,
            frame_rate=0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (25, Fraction(25, 1)),
        (24.0, Fraction(24, 1)),
        (23.976, Fraction(24_000, 1001)),
        (23.98, Fraction(24_000, 1001)),
        (59.94, Fraction(60_000, 1001)),
        (12.5, Fraction(25, 2)),
        (Fraction(30000, 1001), Fraction(30000, 1001)),
    ],
)
def test_frame_rate_resolution(value, expected):
    assert to_av_frame_rate(value) == expected


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), "60"])
def test_invalid_frame_rates(value):
    with pytest.raises(ValueError, match="frame rate"):
        to_av_frame_rate(value)


@pytest.mark.parametrize(
    ("width", "height"), [(0, 10), (10, -1), (1.5, 10), (True, 10)]
)
def test_invalid_video_geometry(width, height):
    with pytest.raises(ValueError, match="dimensions"):
        resolve_video_encoder(
            _output(OutputFormat.MP4),
            width=width,
            height=height,
            frame_rate=30,
        )


def test_custom_options_overlay_defaults_and_are_sorted():
    spec = resolve_video_encoder(
        _output(OutputFormat.MP4),
        width=1920,
        height=1080,
        frame_rate=30,
        codec="libx264",
        options={"preset": "slow", "crf": "18"},
    )

    assert spec is not None
    assert spec.options == (("crf", "18"), ("preset", "slow"))


def test_different_explicit_codec_does_not_inherit_default_options():
    spec = resolve_video_encoder(
        _output(OutputFormat.MP4),
        width=1920,
        height=1080,
        frame_rate=30,
        codec="qtrle",
        pixel_format="argb",
    )

    assert spec is not None
    assert spec.options == ()


@pytest.mark.parametrize("option", ["codec", "pixel_format", "width", "rate"])
def test_stream_fields_cannot_be_supplied_as_codec_options(option):
    with pytest.raises(ValueError, match="explicit stream settings"):
        resolve_video_encoder(
            _output(OutputFormat.MP4),
            width=1920,
            height=1080,
            frame_rate=30,
            options={option: "value"},
        )


def test_encoder_options_must_be_strings():
    with pytest.raises(TypeError, match="keys and values must be strings"):
        resolve_video_encoder(
            _output(OutputFormat.MP4),
            width=1920,
            height=1080,
            frame_rate=30,
            options={"crf": 18},  # type: ignore[dict-item]
        )


def test_unknown_encoder_is_rejected():
    with pytest.raises(ValueError, match="Unknown video encoder"):
        resolve_video_encoder(
            _output(OutputFormat.MP4),
            width=1920,
            height=1080,
            frame_rate=30,
            codec="not-a-codec",
        )


def test_encoder_pixel_format_mismatch_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        resolve_video_encoder(
            _output(OutputFormat.MP4),
            width=1920,
            height=1080,
            frame_rate=30,
            codec="qtrle",
            pixel_format="yuv420p",
        )


def test_video_encoder_fingerprint_is_canonical_and_byte_sensitive():
    spec = VideoEncoderSpec(
        container_format="mp4",
        codec="libx264",
        pixel_format="yuv420p",
        width=1920,
        height=1080,
        frame_rate=Fraction(30, 1),
        options=(("preset", "slow"), ("crf", "18")),
    )
    token = video_encoder_fingerprint(spec)

    assert len(token) == 16
    assert (
        video_encoder_fingerprint(
            replace(spec, options=tuple(reversed(spec.options))),
        )
        == token
    )

    changed_specs = [
        replace(spec, container_format="mov"),
        replace(spec, codec="libx265"),
        replace(spec, pixel_format="yuv444p"),
        replace(spec, width=1280),
        replace(spec, height=720),
        replace(spec, frame_rate=Fraction(60, 1)),
        replace(spec, options=(("crf", "19"), ("preset", "slow"))),
    ]
    assert all(video_encoder_fingerprint(changed) != token for changed in changed_specs)
    assert video_encoder_fingerprint(None) == "none"


def test_transparent_output_requires_alpha_pixel_format():
    with pytest.raises(ValueError, match="alpha-bearing"):
        resolve_video_encoder(
            _output(OutputFormat.WEBM, transparent=True),
            width=1920,
            height=1080,
            frame_rate=30,
            pixel_format="yuv420p",
        )
