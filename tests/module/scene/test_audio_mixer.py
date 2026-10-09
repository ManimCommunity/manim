from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from manim.scene.audio_mixer import (
    SAMPLE_RATE,
    _decode,
    _probe_duration,
    _Sound,
    _SoundMix,
)


def write_wav(
    path: Path,
    value: float = 0.25,
    seconds: float = 1.0,
    channels: int = 1,
    rate: int = SAMPLE_RATE,
) -> Path:
    """Write a constant-level 16-bit WAV; constant levels survive resampling."""
    frames = round(seconds * rate)
    samples = np.full(frames * channels, round(value * 32768), dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path


def mix(sounds, intervals, start=0.0, end=None):
    sound_mix = _SoundMix(sounds, intervals)
    end = sound_mix.duration if end is None else end
    blocks = list(sound_mix.blocks(start, end))
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((2, 0))


def samples(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE)


def test_probe_duration_reads_headers(tmp_path):
    assert _probe_duration(write_wav(tmp_path / "a.wav", seconds=0.5)) == 0.5


@pytest.mark.parametrize("name", ["noise.raw", "notes.txt"])
def test_probe_duration_rejects_unreadable_files(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"\x01\x02" * 100)
    with pytest.raises(ValueError, match="Could not read audio"):
        _probe_duration(path)


def test_decode_copies_mono_to_both_channels_at_full_level(tmp_path):
    decoded = _decode(write_wav(tmp_path / "mono.wav", value=0.5))
    assert decoded.shape == (2, SAMPLE_RATE)
    np.testing.assert_allclose(decoded[:, 100:-100], 0.5, atol=1e-4)


def test_decode_resamples_to_the_mix_rate(tmp_path):
    decoded = _decode(write_wav(tmp_path / "cd.wav", channels=2, rate=44_100))
    assert decoded.shape[0] == 2
    assert abs(decoded.shape[1] - SAMPLE_RATE) <= 64
    np.testing.assert_allclose(decoded[:, 1000:-1000], 0.25, atol=1e-3)


def test_sound_starts_at_its_sample(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", seconds=0.25)
    track = mix([_Sound(beep, 0.5)], [(0.0, 1.0)])

    assert track.shape == (2, SAMPLE_RATE)
    assert np.all(track[:, : samples(0.5)] == 0)
    np.testing.assert_allclose(track[:, samples(0.5) : samples(0.75)], 0.25)
    assert np.all(track[:, samples(0.75) :] == 0)


def test_overlapping_sounds_add_and_gain_is_in_decibels(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", value=0.25)
    track = mix(
        [_Sound(beep, 0.0), _Sound(beep, 0.0, gain=-6.0206)],
        [(0.0, 1.0)],
    )
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


def test_mix_is_clipped(tmp_path):
    loud = write_wav(tmp_path / "loud.wav", value=0.75)
    track = mix([_Sound(loud, 0.0), _Sound(loud, 0.0)], [(0.0, 1.0)])
    assert track.max() == 1.0


def test_excluded_intervals_cut_sounds_like_the_video(tmp_path):
    long = write_wav(tmp_path / "long.wav", seconds=3.0)
    # Scene seconds [1, 2) are excluded; the sound runs through the cut.
    track = mix([_Sound(long, 0.5)], [(0.0, 1.0), (2.0, 3.0)])

    assert track.shape == (2, samples(2.0))
    assert np.all(track[:, : samples(0.5)] == 0)
    np.testing.assert_allclose(track[:, samples(0.5) :], 0.25)


def test_sounds_inside_excluded_intervals_are_not_heard(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", seconds=0.5)
    track = mix([_Sound(beep, 1.25)], [(0.0, 1.0), (2.0, 3.0)])
    assert np.all(track == 0)


def test_negative_start_cuts_the_head(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", seconds=1.0)
    track = mix([_Sound(beep, -0.75)], [(0.0, 1.0)])
    np.testing.assert_allclose(track[:, : samples(0.25)], 0.25)
    assert np.all(track[:, samples(0.25) :] == 0)


def test_blocks_of_a_range_match_the_full_track(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", seconds=1.5)
    sounds = [_Sound(beep, 0.3), _Sound(beep, 1.7, gain=-3)]
    intervals = [(0.0, 1.0), (1.5, 4.0)]
    full = mix(sounds, intervals)
    part = mix(sounds, intervals, start=0.8, end=2.6)

    np.testing.assert_array_equal(part, full[:, samples(0.8) : samples(2.6)])


def test_blocks_are_at_most_one_second():
    sound_mix = _SoundMix([], [(0.0, 3.5)])
    lengths = [block.shape[1] for block in sound_mix.blocks(0, sound_mix.duration)]
    assert lengths == [SAMPLE_RATE] * 3 + [SAMPLE_RATE // 2]


def test_each_source_is_decoded_once(tmp_path, monkeypatch):
    from manim.scene import audio_mixer

    beep = write_wav(tmp_path / "beep.wav", seconds=0.1)
    calls = []

    def counting_decode(path):
        calls.append(path)
        return _decode(path)

    monkeypatch.setattr(audio_mixer, "_decode", counting_decode)
    _SoundMix([_Sound(beep, t) for t in (0.0, 0.2, 0.4)], [(0.0, 1.0)])
    assert calls == [beep]


def test_cut_at_end_counts_only_sounds_heard_at_the_end(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", seconds=1.0)
    # Heard at the end of the last interval: 0.5 s are cut.
    assert _SoundMix([_Sound(beep, 2.5)], [(0.0, 1.0), (2.0, 3.0)]).cut_at_end() == (
        pytest.approx(0.5)
    )
    # Entirely after the last visible interval: excluded, not cut by the end.
    assert _SoundMix([_Sound(beep, 3.5)], [(0.0, 3.0)]).cut_at_end() == 0
    assert _SoundMix([_Sound(beep, 0.0)], [(0.0, 3.0)]).cut_at_end() == 0
