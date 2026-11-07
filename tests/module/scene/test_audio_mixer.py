from __future__ import annotations

from functools import partial

import av
import numpy as np
import pytest

from manim.scene.audio_mixer import (
    SAMPLE_RATE,
    _decode,
    _probe_duration,
    _Sound,
    _SoundMix,
)
from tests.helpers import audio

# Sources at the mix rate are not resampled, so levels and positions are exact.
write_wav = partial(audio.write_wav, seconds=1.0, value=0.25, rate=SAMPLE_RATE)


def mix(sounds, intervals):
    sound_mix = _SoundMix(sounds, intervals)
    return np.concatenate(list(sound_mix.blocks(0, sound_mix.duration)), axis=1)


def samples(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE)


def test_decode_copies_mono_to_both_channels_at_full_level(tmp_path):
    # libswresample's mono-to-stereo upmix lowers the level by 3 dB.
    decoded = _decode(write_wav(tmp_path / "mono.wav", value=0.5))
    np.testing.assert_allclose(decoded, 0.5, atol=1e-4)


def test_audio_that_cannot_be_decoded_raises_a_value_error(tmp_path):
    # AAC headers that still give a duration, followed by data that is not audio.
    path = tmp_path / "damaged.aac"
    with av.open(str(path), "w", format="adts") as container:
        stream = container.add_stream("aac", rate=SAMPLE_RATE, layout="mono")
        tone = np.sin(np.arange(SAMPLE_RATE) * 2 * np.pi * 440 / SAMPLE_RATE)
        frame = av.AudioFrame.from_ndarray(
            0.3 * tone[None, :].astype(np.float32), format="fltp", layout="mono"
        )
        frame.sample_rate = SAMPLE_RATE
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            container.mux(packet)
    data = bytearray(path.read_bytes())
    data[1000:] = np.random.default_rng(0).bytes(len(data) - 1000)
    path.write_bytes(bytes(data))
    assert _probe_duration(path) is not None
    with pytest.raises(ValueError, match="Could not read audio"):
        _decode(path)


def test_a_sound_file_is_probed_again_only_after_it_changes(tmp_path, monkeypatch):
    path = write_wav(tmp_path / "beep.wav", seconds=0.5)
    opened = []
    real_open = av.open
    monkeypatch.setattr(
        "manim.scene.audio_mixer.av.open",
        lambda *args, **kwargs: opened.append(args[0]) or real_open(*args, **kwargs),
    )
    assert _probe_duration(path) == pytest.approx(0.5)
    assert _probe_duration(path) == pytest.approx(0.5)
    assert len(opened) == 1
    write_wav(path, seconds=0.25)
    assert _probe_duration(path) == pytest.approx(0.25)
    assert len(opened) == 2


def test_gain_is_in_decibels(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", value=0.25)
    track = mix([_Sound(beep, 0.0), _Sound(beep, 0.0, gain=-6.0206)], [(0.0, 1.0)])
    np.testing.assert_allclose(track, 0.25 + 0.125, atol=1e-4)


def test_gain_to_background_only_affects_earlier_sounds(tmp_path):
    first = write_wav(tmp_path / "first.wav", value=0.4)
    second = write_wav(tmp_path / "second.wav", value=0.2, seconds=0.5)
    track = mix(
        [_Sound(first, 0.0), _Sound(second, 0.25, gain_to_background=-6.0206)],
        [(0.0, 1.0)],
    )

    np.testing.assert_allclose(track[:, : samples(0.25)], 0.4, atol=1e-4)
    np.testing.assert_allclose(
        track[:, samples(0.25) : samples(0.75)], 0.2 + 0.2, atol=1e-4
    )
    np.testing.assert_allclose(track[:, samples(0.75) :], 0.4, atol=1e-4)


def test_excluded_interval_cuts_a_sound_that_runs_through_it(tmp_path):
    long = write_wav(tmp_path / "long.wav", seconds=3.0)
    # Scene seconds [1, 2) are excluded.
    track = mix([_Sound(long, 0.5)], [(0.0, 1.0), (2.0, 3.0)])

    assert track.shape == (2, samples(2.0))
    assert np.all(track[:, : samples(0.5)] == 0)
    np.testing.assert_allclose(track[:, samples(0.5) :], 0.25)


def test_negative_start_cuts_the_head(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", seconds=1.0)
    track = mix([_Sound(beep, -0.75)], [(0.0, 1.0)])
    np.testing.assert_allclose(track[:, : samples(0.25)], 0.25)
    assert np.all(track[:, samples(0.25) :] == 0)


def test_the_mixer_and_the_writer_place_intervals_at_the_same_movie_times():
    # Both use one scene-to-movie time function; the mix's intervals start
    # exactly where it puts their scene start times, and end where the next begins.
    from manim.scene.audio_mixer import SAMPLE_RATE, _movie_time, _SoundMix

    intervals = [(0.0, 1 / 3), (1.0, 1.0 + 1 / 7), (2.5, 2.5 + 1 / 30)]
    mix = _SoundMix([], intervals)
    for (start, _), (scene, movie, _length) in zip(
        intervals, mix._intervals, strict=True
    ):
        assert movie == round(_movie_time(intervals, start) * SAMPLE_RATE)
        assert scene == round(start * SAMPLE_RATE)
    ends = [movie + length for _, movie, length in mix._intervals]
    assert ends[:-1] == [movie for _, movie, _ in mix._intervals][1:]
    assert mix.duration == _movie_time(intervals, float("inf"))
