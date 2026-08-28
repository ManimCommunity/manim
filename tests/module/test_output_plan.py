from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from manim import __version__
from manim._config.output import OutputFormat, OutputSpec
from manim._config.output_plan import (
    MediaLayoutSpec,
    resolve_file_log_path,
    resolve_media_layout,
    resolve_module_name,
    resolve_output_plan,
    resolve_requested_output_name,
)


def _layout(tmp_path: Path, *, sections: bool = False) -> MediaLayoutSpec:
    root = tmp_path / "not-created"
    return MediaLayoutSpec(
        video_dir=root / "videos",
        images_dir=root / "images",
        sections_dir=root / "sections" if sections else None,
        partial_movie_dir=root / "segments",
        log_dir=root / "logs",
        zero_pad=4,
    )


def _output(
    output_format: OutputFormat,
    *,
    transparent: bool = False,
    save_sections: bool = False,
) -> OutputSpec:
    return OutputSpec(output_format, transparent, save_sections)


@pytest.mark.parametrize(
    ("output_format", "extension"),
    [
        (OutputFormat.MP4, ".mp4"),
        (OutputFormat.MOV, ".mov"),
        (OutputFormat.WEBM, ".webm"),
    ],
)
def test_resolve_video_plan(tmp_path, output_format, extension):
    layout = _layout(tmp_path)

    plan = resolve_output_plan(
        layout,
        _output(output_format),
        scene_name="ExampleScene",
        requested_output_name=None,
    )

    assert plan.primary_artifact == layout.video_dir / f"ExampleScene{extension}"
    assert plan.fallback_image == (
        layout.images_dir / f"ExampleScene_ManimCE_v{__version__}.png"
    )
    assert plan.segment_cache_dir == layout.partial_movie_dir
    assert plan.segment_path("cache-key") == (
        layout.partial_movie_dir / f"cache-key{extension}"
    )
    assert plan.concat_manifest == (
        layout.partial_movie_dir / "partial_movie_file_list.txt"
    )
    assert plan.subcaption_file == layout.video_dir / "ExampleScene.srt"


@pytest.mark.parametrize(
    ("transparent", "segment_extension"),
    [(False, ".mp4"), (True, ".mov")],
)
def test_resolve_gif_plan(tmp_path, transparent, segment_extension):
    layout = _layout(tmp_path)

    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.GIF, transparent=transparent),
        scene_name="ExampleScene",
        requested_output_name=None,
    )

    assert plan.primary_artifact == (
        layout.video_dir / f"ExampleScene_ManimCE_v{__version__}.gif"
    )
    assert plan.segment_extension == segment_extension
    assert plan.segment_path("hash") == layout.partial_movie_dir / (
        f"hash{segment_extension}"
    )


def test_resolve_png_plan(tmp_path):
    layout = _layout(tmp_path)

    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.PNG),
        scene_name="ExampleScene",
        requested_output_name=None,
    )

    assert plan.primary_artifact == (
        layout.images_dir / f"ExampleScene_ManimCE_v{__version__}.png"
    )
    assert plan.fallback_image is None
    with pytest.raises(ValueError, match="does not contain an image sequence"):
        plan.image_frame_path(0)


def test_resolve_png_sequence_plan(tmp_path):
    layout = _layout(tmp_path)

    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.PNG_SEQUENCE),
        scene_name="ExampleScene",
        requested_output_name=None,
    )

    assert plan.primary_artifact == layout.images_dir / "ExampleScene"
    assert plan.image_sequence_dir == layout.images_dir / "ExampleScene"
    assert plan.image_frame_path(0) == layout.images_dir / "ExampleScene" / "0000.png"
    assert plan.image_frame_path(42) == (
        layout.images_dir / "ExampleScene" / "0042.png"
    )


def test_resolve_no_output_plan_without_layout_directories(tmp_path):
    layout = MediaLayoutSpec(None, None, None, None, None, zero_pad=4)

    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.NONE),
        scene_name="ExampleScene",
        requested_output_name=None,
    )

    assert plan.primary_artifact is None
    assert plan.fallback_image is None
    assert plan.segment_cache_dir is None
    assert plan.image_sequence_dir is None
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize(
    ("requested_name", "expected_name"),
    [
        ("movie", "movie.mp4"),
        ("movie.mp4", "movie.mp4"),
        ("movie.mov", "movie.mp4"),
    ],
)
def test_resolved_format_controls_custom_output_suffix(
    tmp_path,
    requested_name,
    expected_name,
):
    layout = _layout(tmp_path)

    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.MP4),
        scene_name="ExampleScene",
        requested_output_name=Path(requested_name),
    )

    assert plan.primary_artifact == layout.video_dir / expected_name
    assert plan.output_stem == "movie"


def test_absolute_output_name_only_relocates_primary_and_fallback(tmp_path):
    layout = _layout(tmp_path, sections=True)
    requested = tmp_path / "exports" / "movie.mov"

    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.MP4, save_sections=True),
        scene_name="ExampleScene",
        requested_output_name=requested,
    )

    assert plan.primary_artifact == tmp_path / "exports" / "movie.mp4"
    assert plan.fallback_image == tmp_path / "exports" / "movie.png"
    assert plan.section_index == layout.sections_dir / "movie.json"
    assert plan.section_path(0, "intro") == (
        layout.sections_dir / "movie_0000_intro.mp4"
    )
    assert plan.segment_cache_dir == layout.partial_movie_dir


def test_nested_output_name_keeps_sections_in_configured_directory(tmp_path):
    layout = _layout(tmp_path, sections=True)

    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.MP4, save_sections=True),
        scene_name="ExampleScene",
        requested_output_name=Path("exports/movie.mp4"),
    )

    assert plan.primary_artifact == layout.video_dir / "exports" / "movie.mp4"
    assert plan.section_path(3, "ending") == (
        layout.sections_dir / "movie_0003_ending.mp4"
    )


def test_plan_resolution_does_not_create_directories(tmp_path):
    layout = _layout(tmp_path, sections=True)
    missing_root = tmp_path / "not-created"

    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.MP4, save_sections=True),
        scene_name="ExampleScene",
        requested_output_name=None,
    )

    assert plan.primary_artifact is not None
    assert not missing_root.exists()


def test_plans_are_immutable_hashable_values(tmp_path):
    layout = _layout(tmp_path)
    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.MP4),
        scene_name="ExampleScene",
        requested_output_name=None,
    )

    assert hash(layout)
    assert hash(plan)
    with pytest.raises(FrozenInstanceError):
        plan.primary_artifact = tmp_path / "other.mp4"


@pytest.mark.parametrize("scene_name", ["", None])
def test_scene_name_is_required(tmp_path, scene_name):
    with pytest.raises(ValueError, match="scene name"):
        resolve_output_plan(
            _layout(tmp_path),
            _output(OutputFormat.MP4),
            scene_name=scene_name,
            requested_output_name=None,
        )


def test_dynamic_path_methods_validate_inputs(tmp_path):
    layout = _layout(tmp_path, sections=True)
    plan = resolve_output_plan(
        layout,
        _output(OutputFormat.MP4, save_sections=True),
        scene_name="ExampleScene",
        requested_output_name=None,
    )

    with pytest.raises(ValueError, match="cache key"):
        plan.segment_path("../escape")
    with pytest.raises(ValueError, match="non-negative"):
        plan.section_path(-1, "intro")


def test_config_adapter_captures_exact_required_directories(config, tmp_path):
    config.media_dir = "relative-media"
    config.input_file = tmp_path / "source" / "example.py"
    config.pixel_height = 480
    config.frame_rate = 15
    config.zero_pad = 3
    config.log_to_file = True
    output = _output(OutputFormat.MP4, save_sections=True)

    module_name = resolve_module_name(config)
    layout = resolve_media_layout(
        config,
        output,
        module_name=module_name,
        scene_name="ExampleScene",
        working_directory=tmp_path,
    )

    quality_dir = tmp_path / "relative-media" / "videos" / "example" / "480p15"
    assert module_name == "example"
    assert layout.video_dir == quality_dir
    assert layout.images_dir == tmp_path / "relative-media" / "images" / "example"
    assert layout.sections_dir == quality_dir / "sections"
    assert layout.partial_movie_dir == (
        quality_dir / "partial_movie_files" / "ExampleScene"
    )
    assert layout.log_dir == tmp_path / "relative-media" / "logs"
    assert layout.zero_pad == 3
    assert not (tmp_path / "relative-media").exists()


def test_config_adapter_skips_unused_output_directories(config, tmp_path):
    config.media_dir = "unused-media"
    config.log_to_file = False

    layout = resolve_media_layout(
        config,
        _output(OutputFormat.NONE),
        module_name="",
        scene_name="ExampleScene",
        working_directory=tmp_path,
    )

    assert layout == MediaLayoutSpec(None, None, None, None, None, zero_pad=4)


def test_requested_name_and_log_path_resolution(config, tmp_path):
    config.output_file = "exports/movie.mp4"
    config.log_to_file = True
    config.media_dir = tmp_path
    module_name = resolve_module_name(config)
    layout = resolve_media_layout(
        config,
        _output(OutputFormat.NONE),
        module_name=module_name,
        scene_name="ExampleScene",
        working_directory=tmp_path,
    )

    assert resolve_requested_output_name(config) == Path("exports/movie.mp4")
    assert (
        resolve_file_log_path(
            layout,
            module_name=module_name,
            scene_name="ExampleScene",
        )
        == tmp_path / "logs" / "_ExampleScene.log"
    )
