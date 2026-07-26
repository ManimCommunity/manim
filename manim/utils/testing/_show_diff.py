from __future__ import annotations

import logging
import warnings

import numpy as np

from manim.typing import PixelArray
from manim.utils.color import BLACK, PURE_GREEN, PURE_RED

__all__ = ["show_diff_helper"]
FRAME_ABSOLUTE_TOLERANCE = 1.01


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

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(frame_data)
    ax1.set_title("Generated")

    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax2.imshow(expected_frame_data)
    ax2.set_title("Expected")

    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
    generated_is_expected = (frame_data == expected_frame_data).all(2)
    expected_is_black = (expected_frame_data == BLACK.to_int_rgba()).all(2)
    diff_im = expected_frame_data.copy()
    diff_im[generated_is_expected & ~expected_is_black] = PURE_GREEN.to_int_rgba()
    diff_im[~generated_is_expected] = PURE_RED.to_int_rgba()
    ax3.imshow(diff_im, interpolation="nearest")
    ax3.set_title("Difference")

    fig.text(
        x=0.55,
        y=0.46,
        s=f"Testname:\n {control_data_filename[:-4]}",
        wrap=True,
        transform=fig.transFigure,
        fontsize=12,
        verticalalignment="top",
    )
    number_of_matches = np.isclose(
        frame_data, expected_frame_data, atol=FRAME_ABSOLUTE_TOLERANCE
    ).sum()
    number_of_mismatches = frame_data.size - number_of_matches
    fig.text(
        x=0.55,
        y=0.34,
        s=f"Difference count:\n {number_of_mismatches}",
        transform=fig.transFigure,
        fontsize=12,
        verticalalignment="top",
    )
    fig.text(
        x=0.55,
        y=0.22,
        s="Difference summary: \n  green = same\n  red = different",
        transform=fig.transFigure,
        fontsize=12,
        verticalalignment="top",
    )

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
