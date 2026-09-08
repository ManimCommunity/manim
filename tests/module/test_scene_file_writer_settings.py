from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

import manim
from manim import Scene, config, tempconfig
from manim.scene import scene_file_writer as writer_module


def test_scene_resolves_consistent_writer_settings_with_empty_assets_root():
    with tempconfig(
        {
            "format": "none",
            "assets_dir": "",
            "max_inflight_encoders": 3,
            "encoder_queue_size": 5,
            "max_files_cached": -1,
        },
    ):
        scene = Scene()
        settings = scene.file_writer_settings

        assert scene.renderer.file_writer.settings is settings
        assert settings.plan is scene.output_plan
        assert settings.plan.output is scene.session_spec.output
        assert settings.video_encoder is scene.session_spec.video_encoder
        assert settings.max_inflight_encoders == 3
        assert settings.encoder_queue_size == 5
        assert settings.max_files_cached == -1
        assert settings.assets_dir == Path.cwd().absolute()


def test_writer_settings_are_private_and_not_exported_from_top_level_manim():
    assert not hasattr(writer_module, "SceneFileWriterSettings")
    assert not hasattr(manim, "SceneFileWriterSettings")
    assert not hasattr(manim, "_SceneFileWriterSettings")


def test_writer_settings_reject_plan_output_mismatches():
    with tempconfig({"format": "mp4"}):
        settings = Scene().file_writer_settings

    mismatched_plan = replace(settings.plan, segment_extension=".webm")
    with pytest.raises(ValueError, match="plan segment extension"):
        replace(settings, plan=mismatched_plan)

    assert settings.video_encoder is not None
    mismatched_encoder = replace(settings.video_encoder, container_format="webm")
    with pytest.raises(ValueError, match="encoder container"):
        replace(settings, video_encoder=mismatched_encoder)


def test_writer_uses_captured_assets_root(tmp_path, monkeypatch):
    captured_assets = tmp_path / "captured"
    changed_assets = tmp_path / "changed"
    captured_assets.mkdir()
    changed_assets.mkdir()
    sound_path = captured_assets / "tone.wav"
    sound_path.touch()

    with tempconfig({"format": "none", "assets_dir": captured_assets}):
        scene = Scene()
        writer = scene.renderer.file_writer
        config.assets_dir = changed_assets

        decoded = Mock()
        from_file = Mock(return_value=decoded)
        monkeypatch.setattr(writer_module.AudioSegment, "from_file", from_file)
        writer.add_audio_segment = Mock()

        writer.add_sound("tone")

    from_file.assert_called_once_with(sound_path)
    writer.add_audio_segment.assert_called_once_with(decoded, None)
