"""Sound-related utility functions."""

from __future__ import annotations

__all__ = [
    "get_full_sound_file_path",
    "read_audio",
    "create_silent_audio",
    "mix_audio",
    "save_audio_to_wav",
]

from typing import TYPE_CHECKING

import av
import numpy as np

from .. import config
from ..utils.file_ops import seek_full_path_from_defaults

if TYPE_CHECKING:
    from pathlib import Path

    from manim.typing import StrPath


# Still in use by add_sound() function in scene_file_writer.py
def get_full_sound_file_path(sound_file_name: StrPath) -> Path:
    """Get the full path to a sound file."""
    return seek_full_path_from_defaults(
        sound_file_name,
        default_dir=config.get_dir("assets_dir"),
        extensions=[".wav", ".mp3"],
    )


def read_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file and return a numpy array of samples and the sample rate."""
    with av.open(str(file_path)) as container:
        stream = container.streams.audio[0]
        sample_rate = stream.sample_rate
        samples = []
        for frame in container.decode(stream):
            frame_samples = frame.to_ndarray()
            # Convert mono to stereo
            if frame_samples.shape[-1] == 1:
                frame_samples = np.repeat(frame_samples, 2, axis=-1)
            samples.append(frame_samples)

        if not samples:
            return np.array([]), sample_rate

        # Find the maximum length among all frames
        max_length = max(frame.shape[0] for frame in samples)
        # Pad each frame to the max length (with zeros)
        padded_samples = []
        for frame in samples:
            if frame.shape[0] < max_length:
                pad_width = ((0, max_length - frame.shape[0]), (0, 0))
                frame = np.pad(frame, pad_width, mode="constant")
            padded_samples.append(frame)

        return np.concatenate(padded_samples, axis=0), sample_rate


def create_silent_audio(duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Create a silent audio segment as a numpy array."""
    num_samples = int(duration * sample_rate)
    return np.zeros((num_samples, 2), dtype=np.float32)


def mix_audio(
    base: np.ndarray, overlay: np.ndarray, position: float, sample_rate: int
) -> np.ndarray:
    """Mix (overlay) one audio array onto another at a specific position."""
    start_sample = int(position * sample_rate)
    end_sample = start_sample + overlay.shape[0]
    result = base.copy()
    if result.shape[0] < end_sample:
        # Pad the result if the overlay extends beyond it
        result = np.pad(result, ((0, end_sample - result.shape[0]), (0, 0)))
    result[start_sample:end_sample] += overlay
    # Normalize to prevent clipping
    max_val = np.max(np.abs(result))
    if max_val > 1:
        result = result / max_val
    return result


def save_audio_to_wav(
    audio_array: np.ndarray, sample_rate: int, output_path: Path
) -> None:
    """Save a numpy audio array to a WAV file using PyAV."""
    # Convert from float32 (-1..1) to int16 (-32768..32767)
    audio_int16 = (audio_array * 32767).astype(np.int16)

    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("pcm_s16le", rate=sample_rate)
        stream.layout = "stereo"  # 2 channels

        frame = av.AudioFrame.from_ndarray(audio_int16, format="s16", layout="stereo")
        frame.sample_rate = sample_rate

        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
