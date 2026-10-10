from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(
    path: Path, seconds: float, value: float = 0.5, rate: int = 44_100
) -> Path:
    """Write a constant-level mono 16-bit WAV; constant levels survive resampling."""
    samples = np.full(round(seconds * rate), round(value * 32767), dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return path
