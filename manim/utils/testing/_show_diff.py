from __future__ import annotations

import logging
import warnings

import numpy as np

from manim.typing import PixelArray
from manim.utils.color import BLACK, PURE_GREEN, PURE_RED

__all__ = ["show_diff_helper"]


def show_diff_helper(
    frame_number: int,
    frame_data: PixelArray,
    expected_frame_data: PixelArray,
    control_data_filename: str,
) -> None:
    """Will visually display with matplotlib differences between frame generated and the one expected."""
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure()
    fig.suptitle(f"Test difference summary at frame {frame_number}", fontsize=16)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(frame_data)
    ax.set_title("Generated")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(expected_frame_data)
    ax.set_title("Expected")

    ax = fig.add_subplot(gs[1, :])
    FRAME_DATA_SHAPE = frame_data.shape
    diff_im = expected_frame_data.copy()
    np.putmask(
        diff_im,
        frame_data != BLACK.to_int_rgba(),  # When generated differs from pure black
        PURE_GREEN.to_int_rgba().astype(np.uint8),  # set pixel to green
    )  # Set any non-black pixels to green
    np.putmask(
        diff_im,
        expected_frame_data != frame_data,
        PURE_RED.to_int_rgba().astype(np.uint8),
    )  # Set any different pixels to red
    # Add the green color channel to all color channels
    expected_frame_data = expected_frame_data + expected_frame_data[:, :, 1].repeat(
        4, axis=1
    ).reshape(FRAME_DATA_SHAPE)
    frame_data = frame_data + frame_data[:, :, 1].repeat(4, axis=1).reshape(
        FRAME_DATA_SHAPE
    )
    np.putmask(
        diff_im,
        expected_frame_data != frame_data,
        PURE_RED.to_int_rgba().astype(np.uint8),
    )  # Set any different pixels to red
    # Add the blue color channel to all color channels
    expected_frame_data = expected_frame_data + expected_frame_data[:, :, 2].repeat(
        4, axis=1
    ).reshape(FRAME_DATA_SHAPE)
    frame_data = frame_data + frame_data[:, :, 2].repeat(4, axis=1).reshape(
        FRAME_DATA_SHAPE
    )
    np.putmask(
        diff_im,
        expected_frame_data != frame_data,
        PURE_RED.to_int_rgba().astype(np.uint8),
    )  # Set any different pixels to red

    ax.imshow(diff_im, interpolation="nearest")
    ax.set_title("Difference summary: (green = same, red = different)")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            plt.show()
        except UserWarning:
            filename = f"{control_data_filename[:-4]}-diff.pdf"
            plt.savefig(filename)
            logging.warning(
                "Interactive matplotlib interface not available,"
                f" diff saved to {filename}."
            )
