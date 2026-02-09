from __future__ import annotations

import logging
import warnings

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
    generated_is_expected = (frame_data == expected_frame_data).all(2)
    expected_is_black = (expected_frame_data == BLACK.to_int_rgba()).all(2)
    diff_im = expected_frame_data.copy()
    diff_im[generated_is_expected & ~expected_is_black] = PURE_GREEN.to_int_rgba()
    diff_im[~generated_is_expected & ~expected_is_black] = PURE_RED.to_int_rgba()
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
