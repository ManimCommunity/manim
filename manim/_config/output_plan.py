"""Internal scene-output path planning."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from manim import __version__

from .output import OutputFormat, OutputSpec


class _LayoutConfigSource(Protocol):
    input_file: str | Path
    output_file: str | Path
    log_to_file: bool
    zero_pad: int

    def get_dir(self, key: str, **kwargs: str) -> Path | None: ...


@dataclass(frozen=True, slots=True)
class MediaLayoutSpec:
    """Exact output directories captured for one scene."""

    video_dir: Path | None
    images_dir: Path | None
    sections_dir: Path | None
    partial_movie_dir: Path | None
    log_dir: Path | None
    zero_pad: int

    def __post_init__(self) -> None:
        for path in (
            self.video_dir,
            self.images_dir,
            self.sections_dir,
            self.partial_movie_dir,
            self.log_dir,
        ):
            if path is not None and not path.is_absolute():
                raise ValueError("Media layout paths must be absolute.")
        if not 0 <= self.zero_pad <= 9:
            raise ValueError("PNG zero padding must be between 0 and 9.")


@dataclass(frozen=True, slots=True)
class OutputPlan:
    """Exact paths and dynamic child-name policy for one scene output.

    The plan retains the immutable :class:`OutputSpec` from which it was
    resolved. ``fallback_image`` is present only when automatic video output may
    fall back to a last-frame PNG for a scene without play calls. For video output,
    ``concat_manifest`` is the persistent diagnostic snapshot of the main scene's
    segment order; assembly does not consume it.
    """

    output: OutputSpec
    primary_artifact: Path | None
    fallback_image: Path | None
    image_sequence_dir: Path | None
    segment_cache_dir: Path | None
    sections_dir: Path | None
    section_index: Path | None
    subcaption_file: Path | None
    concat_manifest: Path | None
    output_stem: str
    segment_extension: str | None
    zero_pad: int

    def image_frame_path(self, frame_index: int) -> Path:
        """Return the exact path for one PNG-sequence frame."""
        if self.image_sequence_dir is None:
            raise ValueError("This output plan does not contain an image sequence.")
        if frame_index < 0:
            raise ValueError("Frame indices must be non-negative.")
        return self.image_sequence_dir / f"{frame_index:0{self.zero_pad}d}.png"

    def segment_path(self, cache_key: str) -> Path:
        """Return the exact path for one silent cached video segment."""
        if self.segment_cache_dir is None or self.segment_extension is None:
            raise ValueError("This output plan does not contain video segments.")
        if not cache_key or Path(cache_key).name != cache_key:
            raise ValueError("A cache key must be a non-empty filename component.")
        return self.segment_cache_dir / f"{cache_key}{self.segment_extension}"

    def section_path(self, index: int, name: str) -> Path:
        """Return the exact path for one derived section video."""
        if self.sections_dir is None or self.segment_extension is None:
            raise ValueError("This output plan does not contain section output.")
        if index < 0:
            raise ValueError("Section indices must be non-negative.")
        section_slug = _slugify_section_name(name)
        return self.sections_dir / (
            f"{self.output_stem}_{index:04}_{section_slug}{self.segment_extension}"
        )


def _slugify_section_name(name: str) -> str:
    """Return a safe filename component while preserving Unicode words."""
    if not isinstance(name, str):
        raise TypeError("Section names must be strings.")
    normalized = unicodedata.normalize("NFKC", name)
    return re.sub(r"[^\w]+", "-", normalized).strip("-_") or "section"


def _absolute_lexical(path: Path, working_directory: Path) -> Path:
    if not working_directory.is_absolute():
        raise ValueError("The output planning working directory must be absolute.")
    anchored = path if path.is_absolute() else working_directory / path
    return Path(os.path.normpath(anchored))


def _required_dir(
    config: _LayoutConfigSource,
    key: str,
    *,
    working_directory: Path,
    module_name: str,
    scene_name: str,
) -> Path:
    path = config.get_dir(key, module_name=module_name, scene_name=scene_name)
    if path is None:
        raise ValueError(f"{key} must not be empty for the requested output.")
    return _absolute_lexical(path, working_directory)


def resolve_segment_cache_directory(
    config: _LayoutConfigSource,
    *,
    module_name: str,
    scene_name: str,
    working_directory: Path,
) -> Path:
    """Resolve the segment-cache directory for one scene."""
    return _required_dir(
        config,
        "partial_movie_dir",
        working_directory=working_directory,
        module_name=module_name,
        scene_name=scene_name,
    )


def resolve_module_name(config: _LayoutConfigSource) -> str:
    """Resolve the source module name used by configured directory templates."""
    if not config.input_file:
        return ""
    input_file = config.get_dir("input_file")
    if input_file is None:
        return ""
    return input_file.stem


def resolve_requested_output_name(
    config: _LayoutConfigSource,
) -> Path | None:
    """Resolve the optional user-requested output name without choosing a format."""
    if not config.output_file:
        return None
    output_file = config.get_dir("output_file")
    if output_file is None:
        return None
    return output_file


def resolve_media_layout(
    config: _LayoutConfigSource,
    output: OutputSpec,
    *,
    module_name: str,
    scene_name: str,
    working_directory: Path,
) -> MediaLayoutSpec:
    """Capture exact directories needed by one concrete scene output."""
    images_dir = None
    video_dir = None
    sections_dir = None
    partial_movie_dir = None

    if output.is_still or output.is_image_sequence or output.fallback_to_still:
        images_dir = _required_dir(
            config,
            "images_dir",
            working_directory=working_directory,
            module_name=module_name,
            scene_name=scene_name,
        )
    if output.is_video:
        video_dir = _required_dir(
            config,
            "video_dir",
            working_directory=working_directory,
            module_name=module_name,
            scene_name=scene_name,
        )
        partial_movie_dir = resolve_segment_cache_directory(
            config,
            working_directory=working_directory,
            module_name=module_name,
            scene_name=scene_name,
        )
        if output.save_sections:
            sections_dir = _required_dir(
                config,
                "sections_dir",
                working_directory=working_directory,
                module_name=module_name,
                scene_name=scene_name,
            )

    log_dir = None
    if config.log_to_file:
        log_dir = _required_dir(
            config,
            "log_dir",
            working_directory=working_directory,
            module_name=module_name,
            scene_name=scene_name,
        )

    return MediaLayoutSpec(
        video_dir=video_dir,
        images_dir=images_dir,
        sections_dir=sections_dir,
        partial_movie_dir=partial_movie_dir,
        log_dir=log_dir,
        zero_pad=config.zero_pad,
    )


def _add_artifact_extension(path: Path, extension: str) -> Path:
    if path.suffix == extension:
        return path
    return path.with_suffix(path.suffix + extension)


def _versioned(path: Path) -> Path:
    return path.with_name(f"{path.stem}_ManimCE_v{__version__}{path.suffix}")


def _output_path(root: Path, name: Path, extension: str) -> Path:
    return root / _add_artifact_extension(name, extension)


def resolve_output_plan(
    layout: MediaLayoutSpec,
    output: OutputSpec,
    *,
    scene_name: str,
    requested_output_name: Path | None,
) -> OutputPlan:
    """Resolve all stable artifact and cache paths for one scene."""
    if not scene_name:
        raise ValueError("A scene name is required for output planning.")

    output_name = requested_output_name or Path(scene_name)
    if output_name.name in {"", ".", ".."}:
        raise ValueError("The requested output name must contain a filename.")
    output_stem = output_name.stem

    if output.format is OutputFormat.NONE:
        return OutputPlan(
            output=output,
            primary_artifact=None,
            fallback_image=None,
            image_sequence_dir=None,
            segment_cache_dir=None,
            sections_dir=None,
            section_index=None,
            subcaption_file=None,
            concat_manifest=None,
            output_stem=output_stem,
            segment_extension=None,
            zero_pad=layout.zero_pad,
        )

    default_name = requested_output_name is None
    normalized_png = None
    versioned_png = None
    if output.is_still or output.is_image_sequence or output.fallback_to_still:
        images_dir = layout.images_dir
        if images_dir is None:
            raise ValueError("Image output requires an images directory.")
        normalized_png = _output_path(images_dir, output_name, ".png")
        versioned_png = _versioned(normalized_png) if default_name else normalized_png

    if output.is_still:
        assert versioned_png is not None
        return OutputPlan(
            output=output,
            primary_artifact=versioned_png,
            fallback_image=None,
            image_sequence_dir=None,
            segment_cache_dir=None,
            sections_dir=None,
            section_index=None,
            subcaption_file=None,
            concat_manifest=None,
            output_stem=output_stem,
            segment_extension=None,
            zero_pad=layout.zero_pad,
        )

    if output.is_image_sequence:
        assert normalized_png is not None
        sequence_dir = normalized_png.with_suffix("")
        return OutputPlan(
            output=output,
            primary_artifact=sequence_dir,
            fallback_image=None,
            image_sequence_dir=sequence_dir,
            segment_cache_dir=None,
            sections_dir=None,
            section_index=None,
            subcaption_file=None,
            concat_manifest=None,
            output_stem=output_stem,
            segment_extension=None,
            zero_pad=layout.zero_pad,
        )

    if not output.is_video:
        raise ValueError(f"Unsupported output format: {output.format.value}")
    if layout.video_dir is None or layout.partial_movie_dir is None:
        raise ValueError("Video output requires video and segment-cache directories.")

    artifact_extension = output.artifact_extension
    assert artifact_extension is not None
    primary_artifact = _output_path(
        layout.video_dir,
        output_name,
        artifact_extension,
    )
    if output.is_gif and default_name:
        primary_artifact = _versioned(primary_artifact)

    sections_dir = layout.sections_dir
    if output.save_sections and sections_dir is None:
        raise ValueError("Section output requires a sections directory.")
    section_index = (
        sections_dir / f"{output_stem}.json" if sections_dir is not None else None
    )

    return OutputPlan(
        output=output,
        primary_artifact=primary_artifact,
        fallback_image=versioned_png if output.fallback_to_still else None,
        image_sequence_dir=None,
        segment_cache_dir=layout.partial_movie_dir,
        sections_dir=sections_dir,
        section_index=section_index,
        subcaption_file=primary_artifact.with_suffix(".srt"),
        concat_manifest=layout.partial_movie_dir / "partial_movie_file_list.txt",
        output_stem=output_stem,
        segment_extension=output.segment_extension,
        zero_pad=layout.zero_pad,
    )


def resolve_file_log_path(
    layout: MediaLayoutSpec,
    *,
    module_name: str,
    scene_name: str,
) -> Path | None:
    """Return the exact optional log-file path for one scene."""
    if layout.log_dir is None:
        return None
    return layout.log_dir / f"{module_name}_{scene_name}.log"
