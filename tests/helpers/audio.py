from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SOURCE_RATE = 44_100
"""Sample rate of the sound files tests write: the CD rate, unlike the mixer's
48 kHz, so that reading them resamples, as for most real sound files."""

_FULL_SCALE = np.iinfo(np.int16).max
"""The 16-bit sample value of level 1."""


def write_wav(
    path: Path,
    seconds: float,
    value: float = 0.5,
    rate: int = SOURCE_RATE,
    channels: int = 1,
) -> Path:
    """Write a constant-level 16-bit WAV; constant levels survive resampling."""
    samples = np.full(
        round(seconds * rate) * channels, round(value * _FULL_SCALE), dtype="<i2"
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path
