from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from manim.__main__ import main


def _segment_directory(
    media_dir: Path,
    *,
    module_name: str,
    quality: str,
    scene_name: str,
) -> Path:
    return (
        media_dir
        / "videos"
        / module_name
        / quality
        / "partial_movie_files"
        / scene_name
    )


def test_cache_clear_resolves_each_scene_without_importing_file(tmp_path):
    scene_file = tmp_path / "example.py"
    scene_file.write_text("raise RuntimeError('must not be imported')\n")
    media_dir = tmp_path / "media"

    for scene_name in ("FirstScene", "SecondScene"):
        directory = _segment_directory(
            media_dir,
            module_name="example",
            quality="480p15",
            scene_name=scene_name,
        )
        directory.mkdir(parents=True)
        (directory / "segment.mp4").touch()
        (directory / ".DS_Store").touch()

    result = CliRunner().invoke(
        main,
        [
            "cache",
            "clear",
            "--media-dir",
            str(media_dir),
            "-q",
            "l",
            str(scene_file),
            "FirstScene",
            "SecondScene",
        ],
        prog_name="manim",
    )

    assert result.exception is None, result.output
    for scene_name in ("FirstScene", "SecondScene"):
        directory = _segment_directory(
            media_dir,
            module_name="example",
            quality="480p15",
            scene_name=scene_name,
        )
        assert not (directory / "segment.mp4").exists()
        assert (directory / ".DS_Store").exists()
    assert "Removed 1 cached segment(s) for FirstScene" in result.output
    assert "Removed 1 cached segment(s) for SecondScene" in result.output


def test_cache_clear_applies_config_file_and_path_overrides(tmp_path):
    scene_file = tmp_path / "configured_scene.py"
    scene_file.touch()
    configured_media_dir = tmp_path / "configured-media"
    overridden_media_dir = tmp_path / "overridden-media"
    config_file = tmp_path / "cache.cfg"
    config_file.write_text(f"[CLI]\nmedia_dir = {configured_media_dir}\n")

    configured_directory = _segment_directory(
        configured_media_dir,
        module_name="configured_scene",
        quality="360p12",
        scene_name="ConfiguredScene",
    )
    configured_directory.mkdir(parents=True)
    (configured_directory / "configured.mp4").touch()

    result = CliRunner().invoke(
        main,
        [
            "cache",
            "clear",
            "--config-file",
            str(config_file),
            "--resolution",
            "640,360",
            "--fps",
            "12",
            str(scene_file),
            "ConfiguredScene",
        ],
        prog_name="manim",
    )

    assert result.exception is None, result.output
    assert not (configured_directory / "configured.mp4").exists()

    overridden_directory = _segment_directory(
        overridden_media_dir,
        module_name="configured_scene",
        quality="360p12",
        scene_name="ConfiguredScene",
    )
    overridden_directory.mkdir(parents=True)
    (overridden_directory / "overridden.mp4").touch()

    result = CliRunner().invoke(
        main,
        [
            "cache",
            "clear",
            "--config-file",
            str(config_file),
            "--media-dir",
            str(overridden_media_dir),
            "--resolution",
            "640,360",
            "--fps",
            "12",
            str(scene_file),
            "ConfiguredScene",
        ],
        prog_name="manim",
    )

    assert result.exception is None, result.output
    assert not (overridden_directory / "overridden.mp4").exists()
