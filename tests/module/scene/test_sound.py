from __future__ import annotations

import struct
import wave
from pathlib import Path

from manim import Scene, tempconfig


def test_add_sound(tmpdir):
    # create sound file
    sound_loc = Path(tmpdir, "noise.wav")
    with wave.open(str(sound_loc), "w") as f:
        f.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        for _ in range(22050):  # half a second of sound
            packed_value = struct.pack("h", 14242)
            f.writeframes(packed_value)
            f.writeframes(packed_value)

    scene = Scene()
    scene.add_sound(sound_loc)


def test_add_sound_after_cached_wait(tmpdir):
    """Regression test for https://github.com/ManimCommunity/manim/issues/4751."""
    sound_loc = Path(tmpdir, "noise.wav")
    with wave.open(str(sound_loc), "w") as f:
        f.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        for _ in range(4410):
            f.writeframes(struct.pack("h", 1000))

    sounds_recorded: list[int] = []

    class AudioAfterSilenceScene(Scene):
        def construct(self):
            self.add_sound(str(sound_loc))
            self.wait(0.2)
            self.wait(0.2)
            self.add_sound(str(sound_loc))
            sounds_recorded.append(len(self.renderer.file_writer.audio_segment))

    with tempconfig({"media_dir": str(tmpdir), "disable_caching": False}):
        AudioAfterSilenceScene().render()
        AudioAfterSilenceScene().render()

    assert len(sounds_recorded) == 2
    assert sounds_recorded[0] > 0
    assert sounds_recorded[1] == sounds_recorded[0]
