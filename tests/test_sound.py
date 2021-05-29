import struct
import wave
from pathlib import Path

from manim import Scene


def test_add_sound(tmpdir):
    # create sound file
    sound_loc = Path(tmpdir, "noise.wav")
    f = wave.open(str(sound_loc), "w")
    f.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
    for _ in range(22050):  # half a second of sound
        packed_value = struct.pack("h", 14242)
        f.writeframes(packed_value)
        f.writeframes(packed_value)

    f.close()

    scene = Scene()
    scene.add_sound(sound_loc)
