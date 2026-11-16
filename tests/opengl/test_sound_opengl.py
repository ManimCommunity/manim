from __future__ import annotations

from pathlib import Path

from manim import Scene
from tests.helpers.audio import write_wav


def test_add_sound(using_opengl_renderer, tmpdir):
    # create sound file
    sound_loc = write_wav(Path(tmpdir, "noise.wav"), seconds=0.5, channels=2)

    scene = Scene()
    scene.add_sound(sound_loc)
