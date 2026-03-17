from __future__ import annotations

import contextlib
import logging
import warnings
from collections.abc import Generator
from pathlib import Path

import numpy as np

from manim.typing import PixelArray

from ._show_diff import show_diff_helper

FRAME_ABSOLUTE_TOLERANCE = 1.01
FRAME_MISMATCH_RATIO_TOLERANCE = 1e-5

logger = logging.getLogger("manim")


class _FramesTester:
    def __init__(self, file_path: Path, show_diff: bool = False) -> None:
        self._file_path = file_path
        self._show_diff = show_diff
        self._frames: np.ndarray
        self._number_frames: int = 0
        self._frames_compared = 0

    @contextlib.contextmanager
    def testing(self) -> Generator[None, None, None]:
        with np.load(self._file_path) as data:
            self._frames = data["frame_data"]
            # For backward compatibility, when the control data contains only one frame (<= v0.8.0)
            if len(self._frames.shape) != 4:
                self._frames = np.expand_dims(self._frames, axis=0)
            logger.debug(self._frames.shape)
            self._number_frames = np.ma.size(self._frames, axis=0)
            yield
            assert self._frames_compared == self._number_frames, (
                f"The scene tested contained {self._frames_compared} frames, "
                f"when there are {self._number_frames} control frames for this test."
            )

    def check_frame(self, frame_number: int, frame: PixelArray) -> None:
        assert frame_number < self._number_frames, (
            f"The tested scene is at frame number {frame_number} "
            f"when there are {self._number_frames} control frames."
        )
        try:
            np.testing.assert_allclose(
                frame,
                self._frames[frame_number],
                atol=FRAME_ABSOLUTE_TOLERANCE,
                err_msg=f"Frame no {frame_number}. You can use --show_diff to visually show the difference.",
                verbose=False,
            )
            self._frames_compared += 1
        except AssertionError as e:
            number_of_matches = np.isclose(
                frame, self._frames[frame_number], atol=FRAME_ABSOLUTE_TOLERANCE
            ).sum()
            number_of_mismatches = frame.size - number_of_matches
            if number_of_mismatches / frame.size < FRAME_MISMATCH_RATIO_TOLERANCE:
                # we tolerate a small (< 0.001%) amount of pixel value errors
                # in the tests, this accounts for minor OS dependent inconsistencies
                self._frames_compared += 1
                warnings.warn(
                    f"Mismatch of {number_of_mismatches} pixel values in frame {frame_number} "
                    f"against control data in {self._file_path}. Below error threshold, "
                    "continuing...",
                    stacklevel=1,
                )
                return

            if self._show_diff:
                show_diff_helper(
                    frame_number,
                    frame,
                    self._frames[frame_number],
                    self._file_path.name,
                )
            raise e


class _ControlDataWriter(_FramesTester):
    def __init__(self, file_path: Path, size_frame: tuple) -> None:
        self.file_path = file_path
        self.frames = np.empty((0, *size_frame, 4))
        self._number_frames_written: int = 0

    # Actually write a frame.
    def check_frame(self, index: int, frame: PixelArray) -> None:
        frame = frame[np.newaxis, ...]
        self.frames = np.concatenate((self.frames, frame))
        self._number_frames_written += 1

    @contextlib.contextmanager
    def testing(self) -> Generator[None, None, None]:
        yield
        self.save_contol_data()

    def save_contol_data(self) -> None:
        self.frames = self.frames.astype("uint8")
        np.savez_compressed(self.file_path, frame_data=self.frames)
        logger.info(
            f"{self._number_frames_written} control frames saved in {self.file_path}",
        )
