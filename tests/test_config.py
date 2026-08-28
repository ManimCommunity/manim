from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from manim import RIGHT, WHITE, Scene, Square, Tex, Text, Vector, tempconfig
from manim._config.output import OutputFormat
from manim._config.render_session import resolve_render_session
from manim._config.utils import ManimConfig
from manim.cli.render.commands import render
from manim.constants import RendererType
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.mobject.types.vectorized_mobject import VMobject
from manim.renderer.protocol import RendererCapabilities
from tests.assert_utils import assert_dir_exists, assert_dir_filled, assert_file_exists


def _resolve_session(config):
    return resolve_render_session(
        config,
        RendererCapabilities(live_preview=True),
        renderer_name="TestRenderer",
    )


def _resolve_output(config):
    return _resolve_session(config).output


def test_tempconfig(config):
    """Test the tempconfig context manager."""
    original = config.copy()

    with tempconfig({"frame_width": 100, "frame_height": 42}):
        # check that config was modified correctly
        assert config["frame_width"] == 100
        assert config["frame_height"] == 42

        # check that no keys are missing and no new keys were added
        assert set(original.keys()) == set(config.keys())

    # check that the keys are still untouched
    assert set(original.keys()) == set(config.keys())

    # check that config is correctly restored
    for k, v in original.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_allclose(config[k], v)
        else:
            assert config[k] == v


def test_tempconfig_restores_renderer_class_bases(config):
    with tempconfig({"renderer": "opengl"}):
        assert config.renderer == RendererType.OPENGL
        assert issubclass(Vector, OpenGLVMobject)

    assert config.renderer == RendererType.CAIRO
    assert issubclass(Vector, VMobject)
    assert not issubclass(Vector, OpenGLVMobject)
    Vector(RIGHT)


@pytest.mark.parametrize(
    ("format", "expected_file_extension"),
    [
        ("mp4", ".mp4"),
        ("webm", ".webm"),
        ("mov", ".mov"),
        ("gif", ".mp4"),
    ],
)
def test_resolve_segment_extensions(config, format, expected_file_extension):
    config.format = format
    assert _resolve_output(config).segment_extension == expected_file_extension


@pytest.mark.parametrize(
    ("format", "expected_format"),
    [
        ("auto", OutputFormat.MP4),
        ("none", OutputFormat.NONE),
        ("png", OutputFormat.PNG),
        ("png-sequence", OutputFormat.PNG_SEQUENCE),
        ("gif", OutputFormat.GIF),
    ],
)
def test_resolve_session_output(config, format, expected_format):
    config.format = format

    assert _resolve_output(config).format is expected_format


def test_transparent_auto_output_resolves_to_mov(config):
    config.format = "auto"
    config.transparent = True

    assert _resolve_output(config).format is OutputFormat.MOV


def test_explicit_no_output_is_not_dry_run(config):
    config.format = "none"

    session = _resolve_session(config)

    assert session.output.format is OutputFormat.NONE
    assert session.dry_run is False


def test_live_preview_auto_output_resolves_to_none(config):
    config.format = "auto"
    config.live_preview = True

    session = _resolve_session(config)

    assert session.output.format is OutputFormat.NONE
    assert session.presentation.live_preview is True
    assert session.dry_run is False


def test_live_preview_requires_renderer_capability(config):
    config.live_preview = True

    with pytest.raises(ValueError, match="does not support live preview"):
        resolve_render_session(
            config,
            RendererCapabilities(),
            renderer_name="TestRenderer",
        )


def test_preview_requires_output(config):
    config.format = "none"
    config.preview = True

    with pytest.raises(ValueError, match="requires a media artifact"):
        resolve_render_session(
            config,
            RendererCapabilities(),
            renderer_name="TestRenderer",
        )


def test_explicit_transparent_mp4_is_rejected(config):
    config.format = "mp4"
    config.transparent = True

    with pytest.raises(ValueError, match="does not support an alpha channel"):
        _resolve_output(config)


def test_dry_run_resolves_no_output_without_mutating_output_request(config):
    config.format = "gif"
    config.save_sections = True
    config.dry_run = True

    session = _resolve_session(config)

    assert session.output.format is OutputFormat.NONE
    assert session.output.save_sections is False
    assert session.dry_run is True
    assert config.format == "gif"
    assert config.save_sections is True


def test_save_last_frame_resolves_to_still_output(config):
    config.format = "auto"
    config.save_last_frame = True

    assert _resolve_output(config).format is OutputFormat.PNG


def test_save_last_frame_alias_works_with_tempconfig(config):
    original_format = config.format

    with tempconfig({"save_last_frame": True}):
        assert config.format == "png"
        assert _resolve_output(config).is_still

    assert config.format == original_format


def test_sections_require_video_output(config):
    config.format = "png"
    config.save_sections = True

    with pytest.raises(ValueError, match="Section output requires"):
        _resolve_output(config)


def test_format_is_loaded_from_config_file(tmp_path, config):
    config_file = tmp_path / "output.cfg"
    config_file.write_text("[CLI]\nformat = png-sequence\n")

    config.digest_file(config_file)

    assert config.format == "png-sequence"


def test_cli_distinguishes_preview_from_live_preview(tmp_path):
    scene_file = tmp_path / "scene.py"
    scene_file.write_text("# --jupyter returns before loading this file\n")

    result = CliRunner().invoke(
        render,
        [str(scene_file), "--jupyter", "--preview", "--live-preview"],
        standalone_mode=False,
    )

    assert result.exception is None
    assert result.return_value.preview is True
    assert result.return_value.live_preview is True


def test_opengl_cli_no_longer_disables_automatic_output(tmp_path, config):
    scene_file = tmp_path / "scene.py"
    scene_file.write_text("# --jupyter returns before loading this file\n")
    result = CliRunner().invoke(
        render,
        [str(scene_file), "--jupyter", "--renderer=opengl"],
        standalone_mode=False,
    )
    assert result.exception is None

    config.format = "auto"
    config.digest_args(result.return_value)

    assert config.renderer is RendererType.OPENGL
    assert config.format == "auto"
    assert _resolve_output(config).format is OutputFormat.MP4


def test_absent_cli_output_options_preserve_config_file_values(tmp_path):
    scene_file = tmp_path / "scene.py"
    scene_file.write_text("# --jupyter returns before loading this file\n")
    config_file = tmp_path / "output.cfg"
    config_file.write_text(
        "[CLI]\n"
        "format = webm\n"
        "output_file = configured-name\n"
        "background_opacity = 0.5\n",
    )
    result = CliRunner().invoke(
        render,
        [str(scene_file), "--jupyter"],
        standalone_mode=False,
    )
    assert result.exception is None

    candidate = ManimConfig().digest_file(config_file)
    candidate.digest_args(result.return_value)

    assert candidate.format == "webm"
    assert candidate.output_file == "configured-name"
    assert candidate.transparent is True


class MyScene(Scene):
    def construct(self):
        self.add(Square())
        self.add(Text("Prepare for unforeseen consequencesλ"))
        self.add(Tex(r"$\lambda$"))
        self.wait(1)


def test_transparent(config):
    """Test the 'transparent' config option."""
    config.verbosity = "ERROR"
    config.dry_run = True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [0, 0, 0, 255])

    config.transparent = True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [0, 0, 0, 0])


def test_transparent_by_background_opacity(config, dry_run):
    config.background_opacity = 0.5
    assert config.transparent is True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [0, 0, 0, 127])
    assert config.transparent is True


def test_background_color(config):
    """Test the 'background_color' config option."""
    config.background_color = WHITE
    config.verbosity = "ERROR"
    config.dry_run = True

    scene = MyScene()
    scene.render()
    frame = scene.renderer.get_frame()
    np.testing.assert_allclose(frame[0, 0], [255, 255, 255, 255])


def test_digest_file(tmp_path, config):
    """Test that a config file can be digested programmatically."""
    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            """
            [CLI]
            media_dir = this_is_my_favorite_path
            video_dir = {media_dir}/videos
            sections_dir = {media_dir}/{scene_name}/prepare_for_unforeseen_consequences
            frame_height = 10
            """,
        )
    config.digest_file(tmp_cfg.name)

    assert config.get_dir("media_dir") == Path("this_is_my_favorite_path")
    assert config.get_dir("video_dir") == Path("this_is_my_favorite_path/videos")
    assert config.get_dir("sections_dir", scene_name="test") == Path(
        "this_is_my_favorite_path/test/prepare_for_unforeseen_consequences"
    )


def test_custom_dirs(tmp_path, config):
    config.media_dir = tmp_path
    config.save_sections = True
    config.log_to_file = True
    config.frame_rate = 15
    config.pixel_height = 854
    config.pixel_width = 480
    config.sections_dir = "{media_dir}/test_sections"
    config.video_dir = "{media_dir}/test_video"
    config.partial_movie_dir = "{media_dir}/test_partial_movie_dir"
    config.images_dir = "{media_dir}/test_images"
    config.text_dir = "{media_dir}/test_text"
    config.tex_dir = "{media_dir}/test_tex"
    config.log_dir = "{media_dir}/test_log"

    scene = MyScene()
    scene.render()
    tmp_path = Path(tmp_path)
    assert_dir_filled(tmp_path / "test_sections")
    assert_file_exists(tmp_path / "test_sections/MyScene.json")

    assert_dir_filled(tmp_path / "test_video")
    assert_file_exists(tmp_path / "test_video/MyScene.mp4")

    assert_dir_filled(tmp_path / "test_partial_movie_dir")
    assert_file_exists(tmp_path / "test_partial_movie_dir/partial_movie_file_list.txt")

    # TODO: another example with image output would be nice
    assert_dir_exists(tmp_path / "test_images")

    assert_dir_filled(tmp_path / "test_text")
    assert_dir_filled(tmp_path / "test_tex")
    assert_dir_filled(tmp_path / "test_log")


def test_pixel_dimensions(tmp_path, config):
    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            """
            [CLI]
            pixel_height = 10
            pixel_width = 10
            """,
        )
    config.digest_file(tmp_cfg.name)

    # aspect ratio is set using pixel measurements
    np.testing.assert_allclose(config.aspect_ratio, 1.0)
    # if not specified in the cfg file, frame_width is set using the aspect ratio
    np.testing.assert_allclose(config.frame_height, 8.0)
    np.testing.assert_allclose(config.frame_width, 8.0)


def test_frame_size(tmp_path, config):
    """Test that the frame size can be set via config file."""
    np.testing.assert_allclose(
        config.aspect_ratio, config.pixel_width / config.pixel_height
    )
    np.testing.assert_allclose(config.frame_height, 8.0)

    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            """
            [CLI]
            pixel_height = 10
            pixel_width = 10
            frame_height = 10
            frame_width = 10
            """,
        )
    config.digest_file(tmp_cfg.name)

    np.testing.assert_allclose(config.aspect_ratio, 1.0)
    # if both are specified in the cfg file, the aspect ratio is ignored
    np.testing.assert_allclose(config.frame_height, 10.0)
    np.testing.assert_allclose(config.frame_width, 10.0)


def test_temporary_dry_run(config):
    """Test that tempconfig correctly restores after setting dry_run."""
    assert _resolve_output(config).is_video

    with tempconfig({"dry_run": True}):
        assert not _resolve_output(config).enabled

    assert _resolve_output(config).is_video


def test_dry_run_with_png_format(config, dry_run):
    """Test that there are no exceptions when running a png without output"""
    config.format = "png"
    config.disable_caching = True
    assert config.dry_run is True
    scene = MyScene()
    scene.render()


def test_dry_run_with_png_format_skipped_animations(config, dry_run):
    """Test that there are no exceptions when running a png without output and skipped animations"""
    config.format = "png"
    config.disable_caching = True
    assert config["dry_run"] is True
    scene = MyScene(skip_animations=True)
    scene.render()


def test_tex_template_file(tmp_path):
    """Test that a custom tex template file can be set from a config file."""
    tex_file = Path(tmp_path / "my_template.tex")
    tex_file.write_text("Hello World!")
    with tempfile.NamedTemporaryFile("w", dir=tmp_path, delete=False) as tmp_cfg:
        tmp_cfg.write(
            f"""
            [CLI]
            tex_template_file = {tex_file}
            """,
        )

    custom_config = ManimConfig().digest_file(tmp_cfg.name)

    assert Path(custom_config.tex_template_file) == tex_file
    assert custom_config.tex_template.body == "Hello World!"


def test_from_to_animations_only_first_animation(config):
    config: ManimConfig
    config.from_animation_number = 0
    config.upto_animation_number = 0

    class SceneWithTwoAnimations(Scene):
        def construct(self):
            self.after_first_animation = False
            s = Square()
            self.add(s)
            self.play(s.animate.scale(2))
            self.renderer.update_skipping_status()
            self.after_first_animation = True
            self.play(s.animate.scale(2))

    scene = SceneWithTwoAnimations()
    scene.render()

    assert scene.after_first_animation is False
