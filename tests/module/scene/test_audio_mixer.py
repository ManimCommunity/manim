from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from manim.scene.audio_mixer import SAMPLE_RATE, _decode, _Sound, _SoundMix
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


def test_cut_at_end_ignores_sounds_that_are_excluded_anyway(tmp_path):
    beep = write_wav(tmp_path / "beep.wav", seconds=1.0)
    # Heard at the end of the movie: 0.5 s are cut.
    assert _SoundMix(
        [_Sound(beep, 2.5)], [(0.0, 1.0), (2.0, 3.0)]
    ).cut_at_end() == pytest.approx(0.5)
    # Entirely after the last shown interval, e.g. in a skipped final section.
    assert _SoundMix([_Sound(beep, 3.5)], [(0.0, 3.0)]).cut_at_end() == 0
