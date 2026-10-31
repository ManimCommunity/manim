from __future__ import annotations

import struct
import wave
from pathlib import Path

from manim import Scene


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


def _write_noise(path, seconds):
    with wave.open(str(path), "w") as f:
        f.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
        packed_value = struct.pack("h", 14242)
        f.writeframes(packed_value * 2 * int(44100 * seconds))


def test_stop_sound_truncates_audio(tmpdir):
    sound_loc = Path(tmpdir, "noise.wav")
    _write_noise(sound_loc, 2)

    scene = Scene()
    file_writer = scene.renderer.file_writer
    scene.add_sound(sound_loc)
    scene.add_sound(sound_loc, time_offset=1.5)
    scene.renderer.time = 0.5
    scene.stop_sound()
    assert file_writer.audio_segment.duration_seconds == 0.5

    # sounds added after stopping play in full
    scene.renderer.time = 1
    scene.add_sound(sound_loc)
    assert file_writer.audio_segment.duration_seconds == 3


def test_stop_sound_without_sound_is_noop():
    scene = Scene()
    scene.stop_sound()
    assert not scene.renderer.file_writer.includes_sound
