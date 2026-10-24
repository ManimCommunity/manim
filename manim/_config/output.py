"""Resolved output configuration for one render session."""

from __future__ import annotations

__all__ = ["OutputFormat", "OutputSpec"]

from dataclasses import dataclass
from enum import StrEnum


class OutputFormat(StrEnum):
    """Supported primary render artifacts."""

    NONE = "none"
    AUTO = "auto"
    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"
    GIF = "gif"
    PNG = "png"
    PNG_SEQUENCE = "png-sequence"

    @classmethod
    def parse(cls, value: str | OutputFormat | None) -> OutputFormat:
        """Normalize a compatibility config value."""
        if value is None or value == "":
            return cls.AUTO
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"Invalid output format: {value!r}")
        normalized = value.lower().replace("_", "-")
        return cls(normalized)


_VIDEO_FORMATS = frozenset(
    {OutputFormat.MP4, OutputFormat.WEBM, OutputFormat.MOV, OutputFormat.GIF}
)


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Immutable, validated output intent for one render session.

    ``format`` is concrete: ``AUTO`` is resolved before this object is created.
    ``fallback_to_still`` allows an automatically selected video format to
    produce a last-frame PNG when a scene has no play calls. The extension of
    cached video segments is deliberately separate from the extension of the
    final artifact; GIF output, for example, uses encoded video segments before
    final GIF assembly.
    """

    format: OutputFormat
    transparent: bool
    save_sections: bool
    fallback_to_still: bool

    def __post_init__(self) -> None:
        if self.format is OutputFormat.AUTO:
            raise ValueError("OutputSpec requires a concrete output format.")
        if self.fallback_to_still and not self.is_video:
            raise ValueError("Still-image fallback requires a video output format.")
        if self.transparent and self.format is OutputFormat.MP4:
            raise ValueError(
                "MP4 output does not support an alpha channel. Use --format=mov "
                "or --format=webm for transparent video.",
            )
        if self.save_sections and not self.is_video:
            raise ValueError("Section output requires a video output format.")

    @property
    def enabled(self) -> bool:
        """Whether this session produces a primary media artifact."""
        return self.format is not OutputFormat.NONE

    @property
    def is_video(self) -> bool:
        """Whether the primary artifact is time-based video."""
        return self.format in _VIDEO_FORMATS

    @property
    def is_still(self) -> bool:
        """Whether only the last frame of the scene is written as PNG."""
        return self.format is OutputFormat.PNG

    @property
    def is_image_sequence(self) -> bool:
        """Whether every rendered frame is written as a PNG image."""
        return self.format is OutputFormat.PNG_SEQUENCE

    @property
    def is_gif(self) -> bool:
        return self.format is OutputFormat.GIF

    @property
    def artifact_extension(self) -> str | None:
        """Extension of the requested primary artifact."""
        if self.format is OutputFormat.NONE:
            return None
        if self.format in {OutputFormat.PNG, OutputFormat.PNG_SEQUENCE}:
            return ".png"
        return f".{self.format.value}"

    @property
    def segment_extension(self) -> str:
        """Container extension used for cached rendered video segments."""
        if not self.is_video:
            raise ValueError("Non-video output does not use video segments.")
        if self.format is OutputFormat.GIF:
            return ".mov" if self.transparent else ".mp4"
        extension = self.artifact_extension
        assert extension is not None
        return extension
