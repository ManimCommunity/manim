"""Probe, decode, and mix scene sounds for media assembly.

Sounds are placed at scene times and mixed into one track for the whole scene. A
movie or section video contains the parts of that track that fall into the scene
intervals it shows.
"""

from __future__ import annotations

__all__: list[str] = []

import functools
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

SAMPLE_RATE = 48_000
"""Sample rate of mixed audio, in Hz."""

LAYOUT = "stereo"
"""Channel layout of mixed audio."""

_BLOCK_SIZE = SAMPLE_RATE  # one second of samples per mixed block


@dataclass(frozen=True, slots=True)
class _Sound:
    """A sound placed on the scene clock.

    Attributes
    ----------
    path
        Resolved path of the sound file.
    start
        Scene time in seconds at which the sound starts. May be negative, in which
        case the part before zero is never heard.
    gain
        Gain applied to the sound, in dB.
    gain_to_background
        Gain applied to all previously added sounds while this one plays, in dB.
    """

    path: Path
    start: float
    gain: float | None = None
    gain_to_background: float | None = None


def _probe_duration(path: Path) -> float | None:
    """Return the duration of the first audio stream of ``path`` without decoding.

    Returns ``None`` if the container headers do not state a duration. The result
    is kept while the file is unchanged: a timeline and the file writer both ask
    for it, and scenes often play the same sound file many times.

    Raises
    ------
    ValueError
        If the file cannot be read as media or contains no audio stream.
    """
    try:
        stamp = os.stat(path)
    except OSError as error:
        raise ValueError(f"Could not read audio from {path}: {error}") from error
    return _probe_unchanged(str(path), stamp.st_mtime_ns, stamp.st_size)


@functools.lru_cache(maxsize=1024)
def _probe_unchanged(path: str, mtime_ns: int, size: int) -> float | None:
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError(f"Could not read audio from {path}: no audio stream.")
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return float(container.duration / av.time_base)
            return None
    except av.FFmpegError as error:
        raise ValueError(f"Could not read audio from {path}: {error}") from error


def _decode(path: Path) -> np.ndarray:
    """Decode the first audio stream of ``path`` to float32 stereo samples.

    The result has shape ``(2, n)`` at :data:`SAMPLE_RATE`. Mono sources are copied
    to both channels at full level; sources with more than two channels are
    downmixed by libswresample.

    Raises
    ------
    ValueError
        If the file's audio data cannot be decoded.
    """
    try:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            layout = "mono" if stream.codec_context.channels == 1 else LAYOUT
            resampler = av.AudioResampler(
                format="fltp", layout=layout, rate=SAMPLE_RATE
            )
            chunks = [
                resampled.to_ndarray()
                for frame in container.decode(stream)
                for resampled in resampler.resample(frame)
            ]
        chunks += [resampled.to_ndarray() for resampled in resampler.resample(None)]
    except av.FFmpegError as error:
        raise ValueError(f"Could not read audio from {path}: {error}") from error
    channels = 1 if layout == "mono" else 2
    samples = (
        np.concatenate(chunks, axis=1)
        if chunks
        else np.zeros((channels, 0), dtype=np.float32)
    )
    if channels == 1:
        samples = np.repeat(samples, 2, axis=0)
    return samples.astype(np.float32, copy=False)


def _db_to_factor(gain: float) -> float:
    return float(10 ** (gain / 20))


def _seconds_to_samples(seconds: float) -> int:
    return round(seconds * SAMPLE_RATE)


def _movie_time(intervals: Sequence[tuple[float, float]], scene_time: float) -> float:
    """Return the movie time at which ``scene_time`` is shown.

    ``intervals`` are the ordered, non-overlapping scene-time spans ``(start,
    end)`` that the movie shows, one after the other. Parts of the scene left
    out take no movie time, so the result never decreases and a scene interval
    maps to one contiguous movie interval. The file writer places sections and
    subcaptions with it, and the mixer its intervals' sample boundaries.
    """
    return sum(max(0.0, min(scene_time, end) - start) for start, end in intervals)


class _SoundMix:
    """Mix sounds and cut the mix to the parts of the scene the movie shows.

    Parameters
    ----------
    sounds
        Sounds in insertion order. The order matters for ``gain_to_background``,
        which affects only the sounds added before.
    intervals
        Ordered, non-overlapping scene-time intervals ``(start, end)`` that are
        shown in the movie. They are concatenated: movie time runs through each
        interval in turn.
    """

    def __init__(
        self,
        sounds: Sequence[_Sound],
        intervals: Sequence[tuple[float, float]],
    ) -> None:
        self._sounds = list(sounds)
        decoded: dict[Path, np.ndarray] = {}
        for sound in self._sounds:
            if sound.path not in decoded:
                decoded[sound.path] = _decode(sound.path)
        self._decoded = decoded

        # Each interval in samples: (scene start, movie start, length). An
        # interval ends at the sample where the next one starts, so rounding
        # never creates gaps or overlaps.
        self._intervals: list[tuple[int, int, int]] = []
        for start, end in intervals:
            movie_start = _seconds_to_samples(_movie_time(intervals, start))
            length = _seconds_to_samples(_movie_time(intervals, end)) - movie_start
            self._intervals.append((_seconds_to_samples(start), movie_start, length))
        self.duration = _movie_time(intervals, math.inf)
        """Total movie duration in seconds."""

        # Each sound's audible pieces: (movie start, source start, length).
        self._placements = [self._place(sound) for sound in self._sounds]

    def _place(self, sound: _Sound) -> list[tuple[int, int, int]]:
        start = _seconds_to_samples(sound.start)
        end = start + self._decoded[sound.path].shape[1]
        pieces = []
        for scene_start, movie_start, length in self._intervals:
            low = max(start, scene_start)
            high = min(end, scene_start + length)
            if low < high:
                pieces.append(
                    (movie_start + low - scene_start, low - start, high - low)
                )
        return pieces

    def blocks(self, start: float, end: float) -> Iterator[np.ndarray]:
        """Yield the mix between two movie times as float32 ``(2, n)`` blocks.

        Blocks are at most one second long and clipped to ``[-1, 1]``.
        """
        first, stop = _seconds_to_samples(start), _seconds_to_samples(end)
        for block_start in range(first, stop, _BLOCK_SIZE):
            block_end = min(block_start + _BLOCK_SIZE, stop)
            block = np.zeros((2, block_end - block_start), dtype=np.float32)
            for sound, pieces in zip(self._sounds, self._placements, strict=True):
                samples = self._decoded[sound.path]
                for movie_start, source_start, length in pieces:
                    low = max(movie_start, block_start)
                    high = min(movie_start + length, block_end)
                    if low >= high:
                        continue
                    region = block[:, low - block_start : high - block_start]
                    if sound.gain_to_background is not None:
                        region *= _db_to_factor(sound.gain_to_background)
                    offset = source_start + low - movie_start
                    source = samples[:, offset : offset + high - low]
                    if sound.gain is not None:
                        source = source * _db_to_factor(sound.gain)
                    region += source
            np.clip(block, -1.0, 1.0, out=block)
            yield block
