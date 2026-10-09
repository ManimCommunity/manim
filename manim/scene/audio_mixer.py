"""Probe, decode, and mix scene sounds for media assembly.

Sounds live on the scene clock. The mix is defined once for the whole scene and is
sampled through the scene-time intervals that are visible in an artifact, so every
artifact (movie, section) is a view of the same global mix.
"""

from __future__ import annotations

__all__: list[str] = []

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

    Returns ``None`` if the container headers do not state a duration.

    Raises
    ------
    ValueError
        If the file cannot be read as media or contains no audio stream.
    """
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
    """
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        layout = "mono" if stream.codec_context.channels == 1 else LAYOUT
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=SAMPLE_RATE)
        chunks = [
            resampled.to_ndarray()
            for frame in container.decode(stream)
            for resampled in resampler.resample(frame)
        ]
    chunks += [resampled.to_ndarray() for resampled in resampler.resample(None)]
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


class _SoundMix:
    """Mix sounds and sample the result through the visible scene intervals.

    Parameters
    ----------
    sounds
        Sounds in insertion order. The order matters for ``gain_to_background``,
        which affects only the sounds added before.
    intervals
        Ordered, non-overlapping scene-time intervals ``(start, end)`` that are
        visible in the output. They are concatenated: output time runs through
        each interval in turn.
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

        # Each interval in samples: (scene start, output start, length). Output
        # boundaries are computed from the running output time once and shared by
        # neighbouring intervals, so rounding never creates gaps or overlaps.
        self._intervals: list[tuple[int, int, int]] = []
        output_time = 0.0
        for start, end in intervals:
            output_start = _seconds_to_samples(output_time)
            output_time += end - start
            length = _seconds_to_samples(output_time) - output_start
            self._intervals.append((_seconds_to_samples(start), output_start, length))
        self.duration = output_time
        """Total output duration in seconds."""

        # Each sound's audible pieces: (output start, source start, length).
        self._placements = [self._place(sound) for sound in self._sounds]

    def _place(self, sound: _Sound) -> list[tuple[int, int, int]]:
        start = _seconds_to_samples(sound.start)
        end = start + self._decoded[sound.path].shape[1]
        pieces = []
        for scene_start, output_start, length in self._intervals:
            low = max(start, scene_start)
            high = min(end, scene_start + length)
            if low < high:
                pieces.append(
                    (output_start + low - scene_start, low - start, high - low)
                )
        return pieces

    def cut_at_end(self) -> float:
        """Return how many seconds of audio the end of the output cuts off.

        Only sounds that are heard in the last visible interval count; sounds that
        lie entirely in excluded parts of the scene are not cut by the end.
        """
        if not self._intervals:
            return 0.0
        scene_start, _, length = self._intervals[-1]
        last_end = scene_start + length
        cut = 0
        for sound in self._sounds:
            start = _seconds_to_samples(sound.start)
            end = start + self._decoded[sound.path].shape[1]
            if start < last_end < end:
                cut = max(cut, end - last_end)
        return cut / SAMPLE_RATE

    def blocks(self, start: float, end: float) -> Iterator[np.ndarray]:
        """Yield the mix between two output times as float32 ``(2, n)`` blocks.

        Blocks are at most one second long and clipped to ``[-1, 1]``.
        """
        first, stop = _seconds_to_samples(start), _seconds_to_samples(end)
        for block_start in range(first, stop, _BLOCK_SIZE):
            block_end = min(block_start + _BLOCK_SIZE, stop)
            block = np.zeros((2, block_end - block_start), dtype=np.float32)
            for sound, pieces in zip(self._sounds, self._placements, strict=True):
                samples = self._decoded[sound.path]
                for output_start, source_start, length in pieces:
                    low = max(output_start, block_start)
                    high = min(output_start + length, block_end)
                    if low >= high:
                        continue
                    region = block[:, low - block_start : high - block_start]
                    if sound.gain_to_background is not None:
                        region *= _db_to_factor(sound.gain_to_background)
                    offset = source_start + low - output_start
                    source = samples[:, offset : offset + high - low]
                    if sound.gain is not None:
                        source = source * _db_to_factor(sound.gain)
                    region += source
            np.clip(block, -1.0, 1.0, out=block)
            yield block
