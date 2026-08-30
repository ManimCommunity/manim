from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from manim import __version__
from manim._config.output import OutputFormat, OutputSpec
from manim._config.output_plan import (
    resolve_media_layout,
    resolve_module_name,
    resolve_output_plan,
    resolve_requested_output_name,
)
from manim._config.video_encoder import resolve_video_encoder
from manim.scene.scene_file_writer import SceneFileWriter


def _make_writer(
    config,
    tmp_path: Path,
    output_format: OutputFormat,
    *,
    transparent: bool = False,
    save_sections: bool = False,
    fallback_to_still: bool = False,
    output_file: str | Path = "",
) -> SceneFileWriter:
    config.media_dir = tmp_path
    config.input_file = tmp_path / "nested" / "example.scene.py"
    config.pixel_height = 480
    config.frame_rate = 15
    config.output_file = output_file

    output = OutputSpec(
        output_format,
        transparent,
        save_sections,
        fallback_to_still,
    )
    module_name = resolve_module_name(config)
    layout = resolve_media_layout(
        config,
        output,
        module_name=module_name,
        scene_name="ExampleScene",
        working_directory=Path.cwd(),
    )
    output_plan = resolve_output_plan(
        layout,
        output,
        scene_name="ExampleScene",
        requested_output_name=resolve_requested_output_name(config),
    )
    renderer = Mock()
    renderer.num_plays = 0
    return SceneFileWriter(
        renderer,
        "ExampleScene",
        output,
        output_plan,
        resolve_video_encoder(
            output,
            width=config.pixel_width,
            height=config.pixel_height,
            frame_rate=config.frame_rate,
        ),
    )


@pytest.mark.parametrize(
    ("output_format", "extension"),
    [
        (OutputFormat.MP4, ".mp4"),
        (OutputFormat.MOV, ".mov"),
        (OutputFormat.WEBM, ".webm"),
    ],
)
def test_default_video_paths(config, tmp_path, output_format, extension):
    writer = _make_writer(config, tmp_path, output_format)
    quality_dir = tmp_path / "videos" / "example.scene" / "480p15"

    assert writer.movie_file_path == quality_dir / f"ExampleScene{extension}"
    assert writer.partial_movie_directory == (
        quality_dir / "partial_movie_files" / "ExampleScene"
    )


@pytest.mark.parametrize(
    ("transparent", "segment_extension"),
    [(False, ".mp4"), (True, ".mov")],
)
def test_gif_primary_and_segment_paths(
    config,
    tmp_path,
    transparent,
    segment_extension,
):
    writer = _make_writer(
        config,
        tmp_path,
        OutputFormat.GIF,
        transparent=transparent,
    )
    quality_dir = tmp_path / "videos" / "example.scene" / "480p15"

    assert writer.movie_file_path == (
        quality_dir / f"ExampleScene_ManimCE_v{__version__}.gif"
    )
    assert writer.gif_file_path == (
        quality_dir / f"ExampleScene_ManimCE_v{__version__}.gif"
    )
    writer.add_partial_movie_file("cache-key")
    assert writer.partial_movie_files == [
        str(
            quality_dir
            / "partial_movie_files"
            / "ExampleScene"
            / f"cache-key{segment_extension}"
        )
    ]


def test_default_png_and_automatic_video_fallback_paths(config, tmp_path):
    png_writer = _make_writer(config, tmp_path, OutputFormat.PNG)
    expected = (
        tmp_path
        / "images"
        / "example.scene"
        / f"ExampleScene_ManimCE_v{__version__}.png"
    )

    png_writer.save_image(Image.new("RGBA", (1, 1)))
    assert png_writer.final_file_path == expected

    video_writer = _make_writer(
        config,
        tmp_path,
        OutputFormat.MP4,
        fallback_to_still=True,
    )
    video_writer.save_image(Image.new("RGBA", (1, 1)))
    assert video_writer.final_file_path == expected


def test_png_sequence_path_and_zero_padding(config, tmp_path):
    config.zero_pad = 3
    writer = _make_writer(config, tmp_path, OutputFormat.PNG_SEQUENCE)

    expected_dir = tmp_path / "images" / "example.scene" / "ExampleScene"
    assert writer.image_sequence_directory == expected_dir

    writer.output_image(Image.new("RGBA", (1, 1)))
    assert (expected_dir / "000.png").is_file()


def test_resolved_output_suffix_preserves_a_different_suffix(config, tmp_path):
    matching = _make_writer(
        config,
        tmp_path,
        OutputFormat.MP4,
        output_file="movie.mp4",
    )
    assert matching.movie_file_path.name == "movie.mp4"

    differing = _make_writer(
        config,
        tmp_path,
        OutputFormat.MP4,
        output_file="movie.mov",
    )
    assert differing.movie_file_path.name == "movie.mov.mp4"


def test_sections_use_configured_directory_for_simple_output_name(config, tmp_path):
    writer = _make_writer(
        config,
        tmp_path,
        OutputFormat.MP4,
        save_sections=True,
        output_file="movie",
    )
    writer.next_section("intro", skip_animations=False, type_="default.normal")

    section = writer.sections[-1]
    assert writer.sections_output_dir == (
        tmp_path / "videos" / "example.scene" / "480p15" / "sections"
    )
    assert section.video == "movie_0000_intro.mp4"
    assert writer.sections_output_dir / section.video == (
        writer.sections_output_dir / "movie_0000_intro.mp4"
    )


def test_nested_and_absolute_output_names_do_not_relocate_sections(
    config,
    tmp_path,
):
    nested = _make_writer(
        config,
        tmp_path,
        OutputFormat.MP4,
        save_sections=True,
        output_file="exports/movie",
    )
    nested.next_section("intro", skip_animations=False, type_="default.normal")
    assert nested.sections_output_dir / nested.sections[-1].video == (
        nested.sections_output_dir / "movie_0000_intro.mp4"
    )

    absolute_name = tmp_path / "exports" / "movie"
    absolute = _make_writer(
        config,
        tmp_path,
        OutputFormat.MP4,
        save_sections=True,
        output_file=absolute_name,
    )
    absolute.next_section("intro", skip_animations=False, type_="default.normal")
    assert not Path(absolute.sections[-1].video).is_absolute()
    assert absolute.sections_output_dir / absolute.sections[-1].video == (
        absolute.sections_output_dir / "movie_0000_intro.mp4"
    )


def test_no_output_plans_no_media_directories(config, tmp_path):
    media_root = tmp_path / "unused-media"
    writer = _make_writer(config, media_root, OutputFormat.NONE)

    assert not hasattr(writer, "movie_file_path")
    assert not hasattr(writer, "image_file_path")
    assert not media_root.exists()
