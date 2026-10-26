"""Reusing cached segments must not change a scene's audio."""

import av
import numpy as np
import pytest

from manim import Scene, Square, tempconfig
from tests.helpers.audio import write_wav


@pytest.fixture
def beep(tmp_path):
    return write_wav(tmp_path / "beep.wav", seconds=0.3)


class Noisy(Scene):
    sound_file = ""

    def construct(self):
        square = Square(fill_opacity=1)
        self.add(square)
        for _ in range(3):
            self.add_sound(self.sound_file)
            self.play(square.animate.shift([0.5, 0.0, 0.0]), run_time=1)


def decoded_audio(path):
    """Decode an artifact's audio stream to mono float samples."""
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
def test_reused_segments_produce_an_identical_audio_track(tmp_path, beep, backend):
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

    # The bug this covers dropped the sounds of reused plays, silencing two of the
    # three beeps.
    assert np.abs(cold).max() > 0.1
    assert warm.size == cold.size
    np.testing.assert_allclose(warm, cold, atol=1e-3)
