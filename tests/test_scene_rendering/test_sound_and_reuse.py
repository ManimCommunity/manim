"""Reusing cached segments must not change a scene's audio."""

import wave

import av
import numpy as np
import pytest

from manim import Scene, Square, tempconfig
from manim.scene.scene_file_writer import SceneFileWriter

# Captured once: a test may call placements() twice, and re-reading the attribute
# would chain each recorder onto the previous one.
_ORIGINAL_ADD_SOUND = SceneFileWriter.add_sound


@pytest.fixture
def beep(tmp_path):
    path = tmp_path / "beep.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(b"\x00\x00" * 22050)
    return path


class Noisy(Scene):
    sound_file = ""

    def construct(self):
        square = Square(fill_opacity=1)
        self.add(square)
        for _ in range(3):
            self.add_sound(self.sound_file)
            self.play(square.animate.shift([0.5, 0.0, 0.0]), run_time=1)


def placements(scene_class, monkeypatch, sound_file):
    """Return the scene times at which sounds reached the writer."""
    recorded = []

    def record(self, path, time, gain=None, **kwargs):
        recorded.append(round(float(time), 3))
        return _ORIGINAL_ADD_SOUND(self, path, time, gain, **kwargs)

    monkeypatch.setattr(SceneFileWriter, "add_sound", record)
    scene = type(
        scene_class.__name__, (scene_class,), {"sound_file": str(sound_file)}
    )()
    scene.render()
    return recorded


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_reused_segments_keep_every_sound(tmp_path, monkeypatch, beep, backend):
    with tempconfig(
        {
            "renderer": backend,
            "format": "mp4",
            "frame_rate": 4,
            "pixel_width": 64,
            "pixel_height": 32,
            "live_preview": False,
            "disable_caching": False,
            "progress_bar": "none",
            "media_dir": str(tmp_path),
        }
    ):
        cold = placements(Noisy, monkeypatch, beep)
        warm = placements(Noisy, monkeypatch, beep)

    assert cold == [0.0, 1.0, 2.0]
    # Before this was fixed, a warm run silently delivered only the first sound.
    assert warm == cold


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def decoded_audio(path):
    """Decode an artifact's audio stream to mono float samples.

    PyAV is used rather than pydub because pydub shells out to the ``ffprobe``
    binary, which is not installed on CI runners.
    """
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        chunks = [frame.to_ndarray() for frame in container.decode(stream)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    samples = np.concatenate(chunks, axis=-1)
    if samples.ndim > 1:
        samples = samples.mean(axis=0)
    return samples.astype(np.float32)


@pytest.mark.parametrize("backend", ["cairo", "opengl"])
def test_reused_segments_produce_an_identical_audio_track(
    tmp_path, monkeypatch, beep, backend
):
    def rendered_audio(media_dir):
        with tempconfig(
            {
                "renderer": backend,
                "format": "mp4",
                "frame_rate": 4,
                "pixel_width": 64,
                "pixel_height": 32,
                "live_preview": False,
                "disable_caching": False,
                "progress_bar": "none",
                "media_dir": str(media_dir),
            }
        ):
            scene = type("Noisy", (Noisy,), {"sound_file": str(beep)})()
            scene.render()
            path = scene.manager.file_writer.final_file_path
        return decoded_audio(path)

    shared = tmp_path / "shared"
    cold = rendered_audio(shared)
    warm = rendered_audio(shared)

    assert cold.size > 0
    # A cache hit must not shorten the track: the bug this covers dropped sounds
    # outright, which removed roughly two thirds of the samples.
    assert warm.size == pytest.approx(cold.size, rel=0.02)
    common = min(cold.size, warm.size)
    np.testing.assert_allclose(warm[:common], cold[:common], atol=1e-3)


def test_excluded_plays_still_drop_sound(tmp_path, monkeypatch, beep):
    """Documented limitation: scene time cannot be placed in a partial artifact.

    Positioning these correctly needs a map from scene time onto the selected
    output span, which is a separate work package.
    """
    with tempconfig(
        {
            "renderer": "cairo",
            "format": "mp4",
            "frame_rate": 4,
            "pixel_width": 64,
            "pixel_height": 32,
            "live_preview": False,
            "disable_caching": True,
            "progress_bar": "none",
            "media_dir": str(tmp_path),
            "from_animation_number": 2,
        }
    ):
        recorded = placements(Noisy, monkeypatch, beep)

    # Only the request made before any play was excluded survives.
    assert recorded == [0.0]


def test_still_output_adds_no_sound(tmp_path, monkeypatch, beep):
    with tempconfig(
        {
            "renderer": "cairo",
            "frame_rate": 4,
            "pixel_width": 64,
            "pixel_height": 32,
            "live_preview": False,
            "disable_caching": True,
            "progress_bar": "none",
            "media_dir": str(tmp_path),
            "save_last_frame": True,
        }
    ):
        recorded = placements(Noisy, monkeypatch, beep)

    assert recorded == [0.0]
