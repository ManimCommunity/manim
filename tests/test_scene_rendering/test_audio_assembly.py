"""Sound in assembled movies: PyAV only, sample-accurate, as long as the video."""

from __future__ import annotations

import av
import numpy as np
import pytest
import srt

from manim import Scene, Square, tempconfig
from manim.scene import scene_file_writer
from tests.helpers.audio import write_wav


def write_float_wav(path, seconds, value=0.5, rate=48_000):
    """Write a 32-bit float WAV; reading one used to require an ffmpeg binary."""
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


def decode_movie(path):
    """Return the video duration, the per-sample audio level, and the audio rate."""
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


def render_scene(tmp_path, scene_class, **config):
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
        scene = scene_class()
        scene.render()
        return scene.manager.file_writer.final_file_path


def render(tmp_path, sounds, **config):
    scene_class = type("Beeps", (Beeps,), {"sounds": sounds})
    return render_scene(tmp_path, scene_class, **config)


def level(levels, rate, start, end):
    """Median level between two movie times, robust to codec ringing."""
    return float(np.median(levels[round(start * rate) : round(end * rate)]))


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
    # No ffmpeg binary may be needed, not even for float WAV files.
    monkeypatch.setenv("PATH", "")
    beep = write_float_wav(tmp_path / "beep.wav", seconds=0.2)
    # Runs two seconds past the end of the one-second video.
    long = write_wav(tmp_path / "long.wav", seconds=3.0, value=0.01)

    path = render(tmp_path, [(str(beep), 0.4), (str(long), 0.0)], **config)
    video_duration, levels, rate = decode_movie(path)

    assert abs(levels.size / rate - video_duration) < 0.03
    onset = np.argmax(levels > 0.2) / rate
    assert onset == pytest.approx(0.4, abs=0.002)


def test_unreadable_sound_fails_at_the_call_site(tmp_path):
    raw = tmp_path / "noise.raw"
    raw.write_bytes(b"\x00\x01" * 1000)
    with pytest.raises(ValueError, match="Could not read audio"):
        render(tmp_path, [(str(raw), 0.0)], format="mp4")


def test_skipped_section_does_not_shift_sound(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", seconds=0.3)

    class SkippedIntro(Scene):
        def construct(self):
            square = Square()
            self.next_section("intro", skip_animations=True)
            self.play(square.animate.shift([1, 0, 0]), run_time=2)
            self.next_section("main")
            self.add_sound(str(beep))  # before the first included play
            self.play(square.animate.shift([-1, 0, 0]), run_time=1)
            self.add_sound(str(beep))
            self.wait(1)

    path = render_scene(tmp_path, SkippedIntro, format="mp4")
    video_duration, levels, rate = decode_movie(path)

    assert video_duration == pytest.approx(2.0, abs=0.07)
    assert level(levels, rate, 0.05, 0.25) == pytest.approx(0.5, abs=0.05)
    assert level(levels, rate, 0.4, 0.9) < 0.01
    assert level(levels, rate, 1.05, 1.25) == pytest.approx(0.5, abs=0.05)
    assert level(levels, rate, 1.4, 1.9) < 0.01


def test_excluded_plays_cut_sound_like_the_video(tmp_path, manim_caplog):
    beep = write_wav(tmp_path / "beep.wav", seconds=0.3)
    # Starts in the excluded first play and continues into the movie.
    long = write_wav(tmp_path / "long.wav", seconds=1.5, value=0.25)

    class ThreePlays(Scene):
        def construct(self):
            square = Square()
            self.add_sound(str(long))
            for _ in range(3):
                self.add_sound(str(beep))
                self.play(square.animate.shift([0.5, 0, 0]), run_time=1)

    path = render_scene(tmp_path, ThreePlays, format="mp4", from_animation_number=1)
    video_duration, levels, rate = decode_movie(path)

    assert video_duration == pytest.approx(2.0, abs=0.07)
    # Beep of play 1 plus the tail of the long sound, then the tail alone.
    assert level(levels, rate, 0.05, 0.25) == pytest.approx(0.75, abs=0.05)
    assert level(levels, rate, 0.35, 0.45) == pytest.approx(0.25, abs=0.05)
    assert level(levels, rate, 0.6, 0.9) < 0.01
    assert level(levels, rate, 1.05, 1.25) == pytest.approx(0.5, abs=0.05)
    # Cuts made by excluding plays are expected and do not warn.
    assert "is cut" not in manim_caplog.text


def test_sound_outside_the_scene_is_cut_with_a_warning(tmp_path, manim_caplog):
    sound = write_wav(tmp_path / "sound.wav", seconds=1.0)

    render(tmp_path, [(str(sound), -0.25), (str(sound), 0.75)], format="mp4")

    assert "starts 0.25 s before the scene starts" in manim_caplog.text
    assert "Sound runs up to 0.75 s past the end of the movie" in manim_caplog.text


def test_subcaptions_follow_the_movie(tmp_path, manim_caplog):
    class Captions(Scene):
        def construct(self):
            square = Square()
            self.next_section("intro", skip_animations=True)
            self.add_subcaption("hidden", duration=1)
            self.play(square.animate.shift([1, 0, 0]), run_time=2)
            self.next_section("main")
            self.add_subcaption("first", duration=0.5)
            self.play(square.animate.shift([-1, 0, 0]), run_time=1)
            self.add_subcaption("last", duration=2)
            self.wait(1)

    path = render_scene(tmp_path, Captions, format="mp4")
    subtitles = list(srt.parse(path.with_suffix(".srt").read_text()))

    assert [
        (s.content, s.start.total_seconds(), s.end.total_seconds()) for s in subtitles
    ] == [
        ("first", 0.0, 0.5),
        ("last", 1.0, 2.0),
    ]
    assert (
        "Subcaption 'last' runs 1.00 s past the end of the movie" in manim_caplog.text
    )


def test_section_videos_carry_their_slice_of_the_mix(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", seconds=0.3)
    long = write_wav(tmp_path / "long.wav", seconds=1.5, value=0.25)

    class TwoSections(Scene):
        def construct(self):
            square = Square()
            self.next_section("first")
            self.add_sound(str(long))  # runs into the second section
            self.play(square.animate.shift([1, 0, 0]), run_time=1)
            self.next_section("second")
            self.add_sound(str(beep), time_offset=0.25)
            self.play(square.animate.shift([-1, 0, 0]), run_time=1)

    with tempconfig({"save_sections": True}):
        path = render_scene(tmp_path, TwoSections, format="mov")
    sections = path.parent / "sections"
    first = decode_movie(next(sections.glob("*first*.mov")))
    second = decode_movie(next(sections.glob("*second*.mov")))
    _, movie, rate = decode_movie(path)

    for video_duration, levels, _ in (first, second):
        assert abs(levels.size / rate - video_duration) < 0.03
    # PCM in mov is lossless, so the slices must match the movie exactly.
    np.testing.assert_array_equal(np.concatenate([first[1], second[1]]), movie)
    # The tail of the long sound, then the beep on top of it.
    assert level(second[1], rate, 0.05, 0.2) == pytest.approx(0.25, abs=0.01)
    assert level(second[1], rate, 0.3, 0.45) == pytest.approx(0.75, abs=0.01)
