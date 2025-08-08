from __future__ import annotations

import logging
import warnings

import numpy as np

from manim.typing import PixelArray


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
    diff_im = expected_frame_data.copy()
    diff_im = np.where(
        frame_data != np.array([0, 0, 0, 255]),
        np.array([0, 255, 0, 255], dtype="uint8"),
        np.array([0, 0, 0, 255], dtype="uint8"),
    )  # Set any non-black pixels to green
    np.putmask(
        diff_im,
        expected_frame_data != frame_data,
        np.array([255, 0, 0, 255], dtype="uint8"),
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
