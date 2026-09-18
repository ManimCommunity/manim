"""Resolved settings for cached video-segment encoding."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import av

from .output import OutputSpec

__all__ = [
    "VideoEncoderSpec",
    "resolve_video_encoder",
    "to_av_frame_rate",
    "video_encoder_fingerprint",
]

_CONFLICTING_OPTION_KEYS = frozenset(
    {
        "codec",
        "codec_name",
        "framerate",
        "height",
        "pix_fmt",
        "pixel_format",
        "rate",
        "video_size",
        "width",
    },
)
_DEFAULT_OPTIONS = {
    "libx264": {"crf": "23"},
    "libvpx-vp9": {"crf": "23"},
    "qtrle": {},
}


@dataclass(frozen=True, slots=True)
class VideoEncoderSpec:
    """Complete byte-affecting settings for one cached video segment."""

    container_format: str
    codec: str
    pixel_format: str
    width: int
    height: int
    frame_rate: Fraction
    options: tuple[tuple[str, str], ...]


def video_encoder_fingerprint(spec: VideoEncoderSpec | None) -> str:
    """Return the stable cache-identity token for resolved encoder settings."""
    if spec is None:
        return "none"
    payload = {
        "schema": "manim-video-segment-v1",
        "container_format": spec.container_format,
        "codec": spec.codec,
        "pixel_format": spec.pixel_format,
        "width": spec.width,
        "height": spec.height,
        "frame_rate": {
            "numerator": spec.frame_rate.numerator,
            "denominator": spec.frame_rate.denominator,
        },
        "options": [
            {"name": name, "value": value} for name, value in sorted(spec.options)
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def to_av_frame_rate(frame_rate: int | float | Fraction) -> Fraction:
    """Return a positive exact rate suitable for a PyAV video stream."""
    if isinstance(frame_rate, bool):
        raise ValueError("frame rate must be a positive finite number")
    if isinstance(frame_rate, Fraction):
        resolved = frame_rate
    elif isinstance(frame_rate, int):
        resolved = Fraction(frame_rate, 1)
    elif isinstance(frame_rate, float) and math.isfinite(frame_rate):
        if abs(frame_rate - round(frame_rate)) < 1e-4:
            resolved = Fraction(round(frame_rate), 1)
        else:
            ntsc_rate = Fraction(round(frame_rate * 1001 / 1000) * 1000, 1001)
            resolved = (
                ntsc_rate
                if abs(frame_rate - float(ntsc_rate)) < 0.02
                else Fraction(str(frame_rate))
            )
    else:
        raise ValueError("frame rate must be a positive finite number")

    if resolved <= 0:
        raise ValueError("frame rate must be positive")
    return resolved


def _default_profile(output: OutputSpec) -> tuple[str, str, str]:
    extension = output.segment_extension
    if extension is None:
        raise ValueError("Video output requires a segment container.")

    container_format = extension.removeprefix(".")
    if extension == ".webm":
        return (
            container_format,
            "libvpx-vp9",
            "yuva420p" if output.transparent else "yuv420p",
        )
    if output.transparent:
        return container_format, "qtrle", "argb"
    return container_format, "libx264", "yuv420p"


def _validate_geometry(width: int, height: int) -> None:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("Video dimensions must be positive integers.")


def _validate_profile(
    *,
    container_format: str,
    codec_name: str,
    pixel_format: str,
    transparent: bool,
) -> None:
    try:
        container = av.format.ContainerFormat(container_format)
    except ValueError as error:
        raise ValueError(f"Unknown output container: {container_format}") from error
    if not container.is_output:
        raise ValueError(f"Container does not support output: {container_format}")

    try:
        codec = av.codec.Codec(codec_name, "w")
    except ValueError as error:
        raise ValueError(f"Unknown video encoder: {codec_name}") from error
    if codec.type != "video":
        raise ValueError(f"Encoder is not a video encoder: {codec_name}")

    try:
        video_format = av.VideoFormat(pixel_format)
    except ValueError as error:
        raise ValueError(f"Unknown pixel format: {pixel_format}") from error

    supported_formats = codec.video_formats
    if supported_formats is not None and pixel_format not in {
        supported.name for supported in supported_formats
    }:
        raise ValueError(
            f"Pixel format {pixel_format} is not supported by encoder {codec_name}.",
        )
    if transparent and not any(
        component.is_alpha for component in video_format.components
    ):
        raise ValueError(
            f"Transparent output requires an alpha-bearing pixel format; "
            f"got {pixel_format}.",
        )


def resolve_video_encoder(
    output: OutputSpec,
    *,
    width: int,
    height: int,
    frame_rate: int | float | Fraction,
    codec: str = "auto",
    pixel_format: str = "auto",
    options: Mapping[str, str] | None = None,
) -> VideoEncoderSpec | None:
    """Resolve and validate cached-segment encoder settings."""
    if not output.is_video:
        return None

    _validate_geometry(width, height)
    resolved_rate = to_av_frame_rate(frame_rate)
    container_format, default_codec, default_pixel_format = _default_profile(output)
    resolved_codec = default_codec if codec == "auto" else codec
    resolved_pixel_format = (
        default_pixel_format if pixel_format == "auto" else pixel_format
    )

    supplied_options = {} if options is None else dict(options)
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in supplied_options.items()
    ):
        raise TypeError("Video encoder option keys and values must be strings.")
    conflicting = sorted(_CONFLICTING_OPTION_KEYS.intersection(supplied_options))
    if conflicting:
        raise ValueError(
            "Video encoder options conflict with explicit stream settings: "
            + ", ".join(conflicting),
        )

    resolved_options = (
        dict(_DEFAULT_OPTIONS[default_codec]) if resolved_codec == default_codec else {}
    )
    resolved_options.update(supplied_options)

    _validate_profile(
        container_format=container_format,
        codec_name=resolved_codec,
        pixel_format=resolved_pixel_format,
        transparent=output.transparent,
    )
    return VideoEncoderSpec(
        container_format=container_format,
        codec=resolved_codec,
        pixel_format=resolved_pixel_format,
        width=width,
        height=height,
        frame_rate=resolved_rate,
        options=tuple(sorted(resolved_options.items())),
    )
