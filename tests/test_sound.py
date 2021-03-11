import os, struct, wave
from pathlib import Path

from manim import Scene


def test_add_sound():
    # create sound file
    f = wave.open("noise.wav", "w")
    f.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
    for _ in range(22050):  # half a second of sound
        packed_value = struct.pack("h", 14242)
        f.writeframes(packed_value)
        f.writeframes(packed_value)

    f.close()

    scene = Scene()
    scene.add_sound("noise.wav")

    os.remove("noise.wav")


def test_add_click_sound():
    # check that the click.mp3 file exists at the right place,
    # this file is needed for the function add_click_sound.
    path_sound = Path(__file__).parent.parent / "manim" / "assets" / "click.mp3"
    assert path_sound.exists() == True
