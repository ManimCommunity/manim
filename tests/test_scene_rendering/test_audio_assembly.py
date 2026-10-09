"""Sound in assembled movies: PyAV only, sample-accurate, as long as the video."""

from __future__ import annotations

import wave

import av
import numpy as np
import pytest

from manim import Scene, Square, tempconfig
from manim.scene import scene_file_writer


def write_wav(path, seconds, value=0.5, rate=44_100):
    data = np.full(round(seconds * rate), round(value * 32767), dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return path


def write_float_wav(path, seconds, value=0.5, rate=48_000):
    """Write a 32-bit float WAV, which pydub could only read through ffmpeg."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_f32le", rate=rate, layout="mono")
        frame = av.AudioFrame.from_ndarray(
            np.full((1, round(seconds * rate)), value, dtype=np.float32),
            format="flt",
            layout="mono",
        )
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def audio_and_video_durations(path):
    """Return (video duration, decoded audio samples as mono, audio rate)."""
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        frames = sum(1 for _ in container.decode(video))
        video_duration = frames / float(video.average_rate)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        chunks = []
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            if not frame.format.is_planar:
                array = array.reshape(-1, frame.layout.nb_channels).T
            levels = np.abs(array.astype(np.float64))
            if array.dtype.kind == "i":
                levels /= np.iinfo(array.dtype).max
            chunks.append(levels.max(axis=0))
        rate = stream.rate
    return video_duration, np.concatenate(chunks), rate


class Beeps(Scene):
    sounds: list[tuple[str, float]] = []

    def construct(self):
        square = Square()
        self.add(square)
        for path, offset in self.sounds:
            self.add_sound(path, time_offset=offset)
        self.play(square.animate.shift([1, 0, 0]), run_time=1)


def render(tmp_path, sounds, **config):
    with tempconfig(
        {
            "frame_rate": 15,
            "pixel_width": 64,
            "pixel_height": 36,
            "progress_bar": "none",
            "media_dir": str(tmp_path / "media"),
            "disable_caching": True,
            **config,
        }
    ):
        scene = type("Beeps", (Beeps,), {"sounds": sounds})()
        scene.render()
        return scene.manager.file_writer.final_file_path


@pytest.mark.parametrize(
    ("config", "codec"),
    [
        ({"format": "mp4"}, "aac"),
        ({"format": "mov"}, "pcm_s16le"),
        ({"format": "webm"}, "libvorbis"),
        ({"format": "webm"}, "libopus"),
    ],
    ids=["mp4", "mov", "webm-vorbis", "webm-opus"],
)
def test_audio_starts_on_time_and_matches_video_length(
    tmp_path, monkeypatch, config, codec
):
    if codec.startswith("lib") and codec not in av.codecs_available:
        pytest.skip(f"PyAV has no {codec} encoder")
    if config.get("format") == "webm":
        monkeypatch.setattr(scene_file_writer, "_webm_audio_codec", lambda: codec)
    beep = write_wav(tmp_path / "beep.wav", seconds=0.2)
    # Runs two seconds past the end of the one-second video.
    long = write_wav(tmp_path / "long.wav", seconds=3.0, value=0.01)

    path = render(tmp_path, [(str(beep), 0.4), (str(long), 0.0)], **config)
    video_duration, levels, rate = audio_and_video_durations(path)

    assert abs(levels.size / rate - video_duration) < 0.03
    onset = np.argmax(levels > 0.2) / rate
    assert onset == pytest.approx(0.4, abs=0.002)


def test_no_ffmpeg_binary_is_needed(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    sound = write_float_wav(tmp_path / "float.wav", seconds=0.5)

    path = render(tmp_path, [(str(sound), 0.0)], format="mp4")
    _, levels, rate = audio_and_video_durations(path)

    assert np.median(levels[: rate // 4]) == pytest.approx(0.5, abs=0.02)


def test_unreadable_sound_fails_at_the_call_site(tmp_path):
    raw = tmp_path / "noise.raw"
    raw.write_bytes(b"\x00\x01" * 1000)
    with pytest.raises(ValueError, match="Could not read audio"):
        render(tmp_path, [(str(raw), 0.0)], format="mp4")
